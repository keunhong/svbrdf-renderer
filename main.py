import numpy as np
from numpy import linalg
import argparse
from vispy import app, gloo
from vispy.gloo import gl
from meshtools import wavefront
from rendtools import (Renderer, Light, SVBRDFMaterial, Renderable,
                       ArcballCamera)
from svbrdf import SVBRDF

app.use_app('glfw')

parser = argparse.ArgumentParser()
parser.add_argument('--brdf', dest='brdf_path', type=str, required=True)
parser.add_argument('--obj', dest='obj_path', type=str, required=True)

args = parser.parse_args()

np.set_printoptions(suppress=True)

class MyRenderer(Renderer):
    def __init__(self, svbrdf, mesh, camera, size):
        super().__init__(size, 0, 1000, camera, show=True)

        gloo.set_state(depth_test=True)
        gloo.set_viewport(0, 0, *self.size)

        self.lights = [
                Light((20, 30, 100), 2000),
                Light((20, 30, -100), 2000),
                Light((0, 100, 10), 2000),
                ]

        vertex_positions = mesh.expand_face_vertices()
        vertex_normals = mesh.expand_face_normals()
        vertex_tangents, vertex_bitangents = mesh.expand_tangents()
        vertex_uvs = mesh.expand_face_uvs()

        material = SVBRDFMaterial(svbrdf)
        self.renderables = [Renderable(material, {
            'a_position': vertex_positions,
            'a_normal': vertex_normals,
            'a_tangent': vertex_tangents,
            'a_bitangent': vertex_bitangents,
            'a_uv': vertex_uvs,
            }, len(self.lights))]

    def update_uniforms(self):
        self.program['cam_pos'] = linalg.inv(self.camera.view_mat())[:3, 3]
        self.program['u_view_mat'] = self.camera.view_mat().T
        self.program['u_model_mat'] = np.eye(4)
        self.program['u_perspective_mat'] = self.camera.perspective_mat().T

        for i, light in enumerate(self.lights):
            self.program['light_position[{}]'.format(i)] = light.position
            self.program['light_intensity[{}]'.format(i)] = light.intensity
            self.program['light_color[{}]'.format(i)] = light.color

    def draw(self):
        gloo.clear(color=(1, 1, 1))
        for renderable in self.renderables:
            self.program = renderable.program
            self.update_uniforms()
            self.program.draw(gl.GL_TRIANGLES)

    def on_key_press(self, event):
        super().on_key_press(event)
        if event.key == '=':
            print('(+) UV scale.')
            for renderable in self.renderables:
                if 'a_uv' in renderable.attributes:
                    renderable.attributes['a_uv'] *= 2
                    renderable.update()
            self.draw()
        elif event.key == '-':
            print('(-) UV scale.')
            for renderable in self.renderables:
                if 'a_uv' in renderable.attributes:
                    renderable.attributes['a_uv'] /= 2
                    renderable.update()
            self.draw()
        self.update()


if __name__=='__main__':
    print('Loading mesh {}'.format(args.obj_path))
    mesh = wavefront.read_obj_file(args.obj_path)
    mesh.resize(100)
    print('Mesh bounding size is {}'.format(mesh.bounding_size()))
    print('Loading BRDF {}'.format(args.brdf_path))
    brdf = SVBRDF(args.brdf_path)

    camera = ArcballCamera(
            size=(1280, 800), fov=75, near=10, far=1000.0,
            position=[0, 0, 120],
            lookat=(0.0, 0.0, -0.0),
            up=(0.0, 1.0, 0.0))

    canvas = MyRenderer(brdf, mesh, camera, size=(1280, 800))
    app.run()
