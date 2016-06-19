import os
import numpy as np
from numpy import linalg
from vispy import gloo, app, util
from vispy.gloo import gl
from . import SVBRDF

_package_dir = os.path.dirname(os.path.realpath(__file__))
_vert_shader_filename = os.path.join(_package_dir, 'svbrdf.vert.glsl')
_frag_shader_filename = os.path.join(_package_dir, 'svbrdf.frag.glsl')

with open(_vert_shader_filename, 'r') as f:
    _vert_shader_source = f.read()
with open(_frag_shader_filename, 'r') as f:
    _frag_shader_source = f.read()


def _normalized(vec):
    return vec / linalg.norm(vec)


def _normalized_to_range(array, lo, hi):
    if hi <= lo:
        raise ValueError('Range must be increasing but {} >= {}.'.format(
            lo, hi))
    min_val = array.min()
    max_val = array.max()
    scale = max_val - min_val if (min_val < max_val) else 1
    return (array - min_val) / scale * (hi - lo) + lo


class Camera:
    def __init__(self, fov, size, near, far, position, lookat, up):
        self.fov = fov
        self.size = size
        self.near = near
        self.far = far

        self.position = np.array(position)
        self.lookat = np.array(lookat)
        self.up = _normalized(np.array(up))

    @property
    def forward(self):
        return _normalized(self.lookat - self.position)

    def perspective_mat(self):
        return util.transforms.perspective(
            self.fov, self.size[0] / self.size[1], self.near, self.far).T

    def view_mat(self):
        rotation_mat = np.eye(3)
        rotation_mat[0, :] = np.cross(self.forward, self.up)
        rotation_mat[1, :] = self.up
        rotation_mat[2, :] = -self.forward

        position = rotation_mat.dot(self.position)

        view_mat = np.eye(4)
        view_mat[:3, :3] = rotation_mat
        view_mat[:3, 3] = -position

        return view_mat


class Canvas(app.Canvas):
    def __init__(self, svbrdf: SVBRDF, size=(800, 600)):
        app.Canvas.__init__(self, size=size, show=True)

        self.svbrdf = svbrdf

        self.program = gloo.Program(_vert_shader_source,
                                    _frag_shader_source)

        self.camera = Camera(
            size=size, fov=75, near=0.1, far=1000.0,
            position=(5.0, 5.0, 30.0),
            lookat=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0))

        self.light_azimuth = 0.0
        self.light_elevation = np.pi / 4
        self.light_distance = 55.0
        self.light_intensity = 1500.0

        self.object_position = (0, 0)
        self.object_scale = 100.0
        self.object_rotation = 0
        self.light_color = (1.0, 1.0, 1.0)

        self.program['u_view_mat'] = self.camera.view_mat().T
        self.program['u_perspective_mat'] = self.camera.perspective_mat().T
        self.program['light_azimuth'] = self.light_azimuth
        self.program['light_elevation'] = self.light_elevation
        self.program['light_distance'] = self.light_distance
        self.program['light_intensity'] = self.light_intensity
        self.program['light_color'] = self.light_color

        self.program['object_position'] = self.object_position
        self.program['object_rotation'] = self.object_rotation

        # self.program['cam_pos'] = -self.camera.position
        self.program['cam_pos'] = linalg.inv(self.camera.view_mat())[:3, 3]

        self.program['alpha'] = self.svbrdf.alpha
        spec_map_tex = gloo.Texture2D(
            _normalized_to_range(self.svbrdf.specular_map, 0, 1),
            interpolation='linear')
        diff_map_tex = gloo.Texture2D(
            self.svbrdf.diffuse_map, interpolation='linear')
        self.program['diff_map_tex'] = diff_map_tex
        self.program['spec_map_tex'] = spec_map_tex
        self.program['spec_shape_map_tex'] = self.svbrdf.spec_shape_map
        self.program['normal_map_tex'] = self.svbrdf.normal_map

        height, width, _ = self.svbrdf.diffuse_map.shape
        aspect = height / width
        planesize = self.object_scale * 0.5
        positions = np.array([
            [-planesize, -planesize * aspect, 0],
            [-planesize, planesize * aspect, 0],
            [planesize, -planesize * aspect, 0],
            [planesize, planesize * aspect, 0],
        ], dtype=np.float32)
        uvs = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ], dtype=np.float32)

        self.program['a_position'] = positions
        self.program['a_uv'] = uvs

        self.timer = app.Timer('auto', self.on_timer)
        self.timer.start()

    def on_draw(self, event):
        gloo.clear('black')
        self.program.draw(gl.GL_TRIANGLE_STRIP)

    def on_timer(self, event):
        self.light_azimuth += 0.05
        self.program['light_azimuth'] = self.light_azimuth
        self.update()

