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
        mat = util.transforms.perspective(
            self.fov, self.size[0] / self.size[1], self.near, self.far).T
        mat[0, 0] *= -1
        mat[1, 1] *= -1
        return mat

    def view_mat(self):
        rotation_mat = np.eye(3)
        rotation_mat[2, :] = _normalized(-self.forward)
        rotation_mat[0, :] = np.cross(self.up, rotation_mat[2, :])
        rotation_mat[1, :] = np.cross(rotation_mat[2, :], rotation_mat[0, :])

        position = rotation_mat.T.dot(self.position)

        view_mat = np.eye(4)
        view_mat[:3, :3] = rotation_mat
        view_mat[:3, 3] = -position
        print(view_mat)

        return view_mat


class Canvas(app.Canvas):
    def __init__(self, svbrdf: SVBRDF, mesh: Mesh, size=(800, 600)):
        app.Canvas.__init__(self, size=size, show=True)
        gloo.set_state(depth_test=True)
        gloo.set_viewport(0, 0, *self.size)

        self.svbrdf = svbrdf

        self.program = gloo.Program(_vert_shader_source,
                                    _frag_shader_source)

        self.camera_rot = 0
        self.camera = Camera(
            size=size, fov=75, near=10, far=300.0,
            position=(5.0, 55.0, 60.0),
            lookat=(0.0, 10.0, -0.0),
            up=(0.0, 0.0, -1.0))

        self.light_intensity = 8200.0

        self.light_color = (1.0, 1.0, 1.0)

        self.program['light_position'] = (90, 90, 90)
        self.program['light_intensity'] = self.light_intensity
        self.program['light_color'] = self.light_color

        self.update_uniforms()

        self.program['alpha'] = self.svbrdf.alpha
        self.program['diff_map'] = Texture2D(self.svbrdf.diffuse_map,
                                             interpolation='linear',
                                             wrapping='repeat')
        self.program['spec_map'] = Texture2D(self.svbrdf.specular_map,
                                             interpolation='linear',
                                             wrapping='repeat')
        self.svbrdf.spec_shape_map[:, :, :] /= 100
        self.program['spec_shape_map'] = self.svbrdf.spec_shape_map

        self.program['normal_map'] = Texture2D(self.svbrdf.normal_map,
                                               interpolation='linear',
                                               wrapping='repeat')

        height, width, _ = self.svbrdf.diffuse_map.shape

        vertex_positions = mesh.expand_face_vertices()
        vertex_normals = mesh.expand_face_normals()
        vertex_tangents, vertex_bitangents = mesh.expand_tangents()
        vertex_uvs = mesh.expand_face_uvs()
        self.program['a_position'] = vertex_positions
        self.program['a_normal'] = vertex_normals
        self.program['a_uv'] = vertex_uvs
        self.program['a_tangent'] = vertex_tangents
        self.program['a_bitangent'] = vertex_bitangents

        self.timer = app.Timer('auto', self.on_timer)
        self.timer.start()

    def update_uniforms(self):
        view_mat = np.array([
            [0.993884, -0.110432, 0.000000, 1.104316],
            [0.088150, 0.793346, -0.602355, -7.933460],
            [0.066519, 0.598671, 0.798228, -81.153198],
            [0.000000, 0.000000, 0.000000, 1.000000]], dtype=np.float32)
        self.program['cam_pos'] = linalg.inv(view_mat)[:3, 3]
        # print(self.program['cam_pos'])
        # self.program['u_view_mat'] = self.camera.view_mat()
        self.program['u_view_mat'] = view_mat.T
        # print(self.camera.perspective_mat())
        # print(self.program['u_view_mat'])
        self.program['u_perspective_mat'] = self.camera.perspective_mat().T

    def on_draw(self, event):
        gloo.clear(color=(1, 1, 1))
        self.program.draw(gl.GL_TRIANGLES)

    def on_timer(self, event):
        # self.light_azimuth += 0.01
        # self.camera_rot += 0.005
        # self.camera.position[0] = 100.0 * np.sin(self.camera_rot)
        # self.camera.position[2] = 100.0 * np.cos(self.camera_rot)
        self.update_uniforms()
        self.update()
