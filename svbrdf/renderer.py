import os
from string import Template

import numpy as np
from numpy import linalg
from vispy import gloo, app, util
from vispy.gloo import gl, Texture2D
from vispy.util.quaternion import Quaternion

from meshtools.mesh import Mesh
import rendtools as rt
from . import SVBRDF

_package_dir = os.path.dirname(os.path.realpath(__file__))
_vert_shader_filename = os.path.join(_package_dir, 'svbrdf.vert.glsl')
_frag_shader_filename = os.path.join(_package_dir, 'svbrdf.frag.glsl')

with open(_vert_shader_filename, 'r') as f:
    _vert_shader_source = Template(f.read())
with open(_frag_shader_filename, 'r') as f:
    _frag_shader_source = Template(f.read())


class Program:
    def __init__(self, vert_shader, frag_shader, lights):
        self.lights = lights
        self.vert_shader = vert_shader.substitute()
        self.frag_shader = frag_shader.substitute(num_lights=len(self.lights))

    def compile(self):
        return gloo.Program(self.vert_shader, self.frag_shader)


class Canvas(app.Canvas):
    def __init__(self, svbrdf: SVBRDF, mesh: Mesh, size=(800, 600)):
        app.Canvas.__init__(self, size=size, show=True)
        gloo.set_state(depth_test=True)
        gloo.set_viewport(0, 0, *self.size)

        self.svbrdf = svbrdf

        self.camera_rot = 0
        self.quaternion = Quaternion()
        self.camera = rt.ArcballCamera(
            size=size, fov=75, near=10, far=1000.0,
            position=[0, 0, 150],
            lookat=(0.0, 0.0, -0.0),
            up=(0.0, 1.0, 0.0))
        print(self.camera.position)

        self.lights = [
            rt.Light((20, 30, 100), 2000),
            rt.Light((20, 30, -100), 2000),
            rt.Light((0, 100, 10), 2000),
        ]

        self.program = Program(_vert_shader_source,
                               _frag_shader_source,
                               self.lights).compile()

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
        self.program['a_uv'] = vertex_uvs * 2
        self.program['a_tangent'] = vertex_tangents
        self.program['a_bitangent'] = vertex_bitangents

    def update_uniforms(self):
        self.program['cam_pos'] = linalg.inv(self.camera.view_mat())[:3, 3]
        self.program['u_view_mat'] = self.camera.view_mat().T
        # self.program['u_model_mat'] = self.model_quat.get_matrix().T
        self.program['u_model_mat'] = np.eye(4)
        self.program['u_perspective_mat'] = self.camera.perspective_mat().T

        for i, light in enumerate(self.lights):
            self.program['light_position[{}]'.format(i)] = light.position
            self.program['light_intensity[{}]'.format(i)] = light.intensity
            self.program['light_color[{}]'.format(i)] = light.color

    def on_resize(self, event):
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        self.camera.size = self.size
        self.update_uniforms()
        self.update()

    def on_draw(self, event):
        gloo.clear(color=(1, 1, 1))
        self.program.draw(gl.GL_TRIANGLES)

    def on_key_press(self, event):
        if event.key == 'Escape':
            self.app.quit()

    def on_mouse_move(self, event):
        if event.is_dragging:
            self.camera.handle_mouse(event.last_event.pos, event.pos)
            self.update_uniforms()
            self.update()
