import os
import numpy as np
from numpy import linalg
from vispy import gloo, app, util
from vispy.gloo import gl, Texture2D
from meshtools.mesh import Mesh
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
    def __init__(self, svbrdf: SVBRDF, mesh : Mesh, size=(800, 600)):
        app.Canvas.__init__(self, size=size, show=True)

        self.svbrdf = svbrdf

        self.program = gloo.Program(_vert_shader_source,
                                    _frag_shader_source)
        gloo.set_cull_face('front_and_back')

        self.camera_rot = 0
        self.camera = Camera(
            size=size, fov=75, near=10, far=500.0,
            position=(55.0, 20.0, 50.0),
            lookat=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0))

        self.light_azimuth = np.pi / 4
        self.light_elevation = np.pi / 2
        self.light_distance = 120.0
        self.light_intensity = 3200.0

        self.object_position = (0, 0)
        self.object_scale = 100.0
        self.object_rotation = 0
        self.light_color = (1.0, 1.0, 1.0)

        self.program['light_azimuth'] = self.light_azimuth
        self.program['light_elevation'] = self.light_elevation
        self.program['light_distance'] = self.light_distance
        self.program['light_intensity'] = self.light_intensity
        self.program['light_color'] = self.light_color

        self.program['object_position'] = self.object_position
        self.program['object_rotation'] = self.object_rotation

        self.update_uniforms()

        self.program['alpha'] = self.svbrdf.alpha
        self.program['diff_map'] = Texture2D(self.svbrdf.diffuse_map,
                                             interpolation='linear',
                                             wrapping='repeat')
        self.program['spec_map'] = Texture2D(self.svbrdf.specular_map,
                                             interpolation='linear',
                                             wrapping='repeat')
        self.program['spec_shape_map'] = Texture2D(self.svbrdf.spec_shape_map,
                                                   interpolation='linear',
                                                   wrapping='repeat')
        self.program['normal_map'] = Texture2D(self.svbrdf.normal_map,
                                                   interpolation='linear',
                                                   wrapping='repeat')

        height, width, _ = self.svbrdf.diffuse_map.shape

        vertex_positions = mesh.expand_face_vertices()
        vertex_normals = mesh.expand_face_normals()
        vertex_tangents, vertex_bitangents = mesh.expand_tangents()
        vertex_uvs = mesh.expand_face_uvs() / 100
        self.program['a_position'] = vertex_positions
        self.program['a_normal'] = vertex_normals
        self.program['a_uv'] = vertex_uvs
        self.program['a_tangent'] = vertex_tangents
        self.program['a_bitangent'] = vertex_bitangents

        self.timer = app.Timer('auto', self.on_timer)
        self.timer.start()

    def update_uniforms(self):
        self.program['cam_pos'] = self.camera.position
        self.program['u_view_mat'] = self.camera.view_mat().T
        self.program['u_perspective_mat'] = self.camera.perspective_mat().T


    def on_draw(self, event):
        gloo.clear('black')
        self.program.draw(gl.GL_TRIANGLES)

    def on_timer(self, event):
        # self.light_azimuth += 0.01
        self.camera_rot += 0.01
        self.camera.position[0] = 75.0 * np.sin(self.camera_rot)
        self.camera.position[2] = 75.0 * np.cos(self.camera_rot)
        self.update_uniforms()
        self.program['light_azimuth'] = self.light_azimuth
        self.update()

