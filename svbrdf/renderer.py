import os

import numpy as np
from numpy import linalg
from vispy import gloo, app, util
from vispy.gloo import gl, Texture2D
from vispy.util.quaternion import Quaternion

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


def _arcball(x, y, w, h):
    """Convert x,y coordinates to w,x,y,z Quaternion parameters
    Adapted from:
    linalg library
    Copyright (c) 2010-2015, Renaud Blanch <rndblnch at gmail dot com>
    Licence at your convenience:
    GPLv3 or higher <http://www.gnu.org/licenses/gpl.html>
    BSD new <http://opensource.org/licenses/BSD-3-Clause>
    """
    r = (w + h) / 2.
    x, y = -(2. * x - w) / r, -(2. * y - h) / r
    h = np.sqrt(x*x + y*y)
    return (0., x/h, y/h, 0.) if h > 1. else (0., x, y, np.sqrt(1. - h*h))


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
        return mat

    def view_mat(self):
        rotation_mat = np.eye(3)
        rotation_mat[0, :] = np.cross(self.forward, self.up)
        rotation_mat[2, :] = -self.forward
        # We recompute the 'up' vector portion of the matrix as the cross
        # product of the forward and sideways vector so that we have an ortho-
        # normal basis.
        rotation_mat[1, :] = np.cross(rotation_mat[2, :], rotation_mat[0, :])

        position = rotation_mat.dot(self.position)

        view_mat = np.eye(4)
        view_mat[:3, :3] = rotation_mat
        view_mat[:3, 3] = -position

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
            size=size, fov=75, near=10, far=1000.0,
            position=(100.0, 10.0, 150.0),
            lookat=(0.0, 0.0, -0.0),
            up=(0.0, 1.0, 0.0))

        self.light_intensity = 2500.0

        self.light_color = (1.0, 1.0, 1.0)

        self.program['light_position'] = (20, 30, 100)
        self.program['light_intensity'] = self.light_intensity
        self.program['light_color'] = self.light_color

        self.update_uniforms()

        self.program['alpha'] = self.svbrdf.alpha
        self.program['diff_map'] = Texture2D(self.svbrdf.diffuse_map,
                                             interpolation='linear',
                                             wrapping='repeat',
                                             internalformat='rgb32f')
        self.program['spec_map'] = Texture2D(self.svbrdf.specular_map,
                                             interpolation='linear',
                                             wrapping='repeat',
                                             internalformat='rgb32f')
        self.program['spec_shape_map'] = Texture2D(self.svbrdf.spec_shape_map,
                                                   internalformat='rgb32f')
        self.program['normal_map'] = Texture2D(self.svbrdf.normal_map,
                                               interpolation='linear',
                                               wrapping='repeat',
                                               internalformat='rgb32f')

        height, width, _ = self.svbrdf.diffuse_map.shape

        vertex_positions = mesh.expand_face_vertices()
        vertex_normals = mesh.expand_face_normals()
        vertex_tangents, vertex_bitangents = mesh.expand_tangents()
        vertex_uvs = mesh.expand_face_uvs()
        print(vertex_uvs.min())
        self.program['a_position'] = vertex_positions
        self.program['a_normal'] = vertex_normals
        self.program['a_uv'] = vertex_uvs
        self.program['a_tangent'] = vertex_tangents
        self.program['a_bitangent'] = vertex_bitangents

    def update_uniforms(self):
        self.program['cam_pos'] = linalg.inv(self.camera.view_mat())[:3, 3]
        self.program['u_view_mat'] = self.camera.view_mat().T
        self.program['u_perspective_mat'] = self.camera.perspective_mat().T

    def on_resize(self, event):
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        self.camera.size = self.size
        self.update_uniforms()
        self.update()

    def on_draw(self, event):
        gloo.clear(color=(1, 1, 1))
        self.program.draw(gl.GL_TRIANGLES)

    def on_mouse_move(self, event):
        if event.is_dragging:
            x0, y0 = event.last_event.pos
            x1, y1 = event.pos
            w, h = self.size
            quat = (Quaternion(*_arcball(x1, y1, w, h))
                    * Quaternion(*_arcball(x0, y0, w, h)))
            self.camera.position = quat.get_matrix()[:3, :3].dot(self.camera.position)
            self.update_uniforms()
            self.update()
