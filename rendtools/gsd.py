import numpy as np
from numpy import linalg
from vispy import gloo
from vispy.gloo import gl

from svbrdf import SVBRDF
from meshtools import wavefront

from . import (Renderer, Renderable, Light, SVBRDFMaterial, PhongMaterial,
               SVBRDFColorTransferMaterial)


class GSDRenderer(Renderer):

    def __init__(self, gsd_dict, camera, size=(800, 600), *args, **kwargs):
        super().__init__(size, 0, 1000, camera, *args, **kwargs)
        gloo.set_state(depth_test=True)
        gloo.set_viewport(0, 0, *self.size)
        self.scene = GSDScene(gsd_dict)

    def update_uniforms(self):
        self.program['cam_pos'] = linalg.inv(self.camera.view_mat())[:3, 3]
        self.program['u_view_mat'] = self.camera.view_mat().T
        self.program['u_model_mat'] = np.eye(4)
        self.program['u_perspective_mat'] = self.camera.perspective_mat().T

        for i, light in enumerate(self.scene.lights):
            self.program['light_position[{}]'.format(i)] = light.position
            self.program['light_intensity[{}]'.format(i)] = light.intensity
            self.program['light_color[{}]'.format(i)] = light.color

    def update_alpha(self, alpha):
        for renderable in self.scene.renderables:
            if type(renderable.material) is SVBRDFMaterial:
                renderable.material.alpha = alpha
                renderable.update()
        self.update()

    def draw(self):
        gloo.clear(color=(1, 1, 1))
        for renderable in self.scene.renderables:
            self.program = renderable.program
            self.update_uniforms()
            self.program.draw(gl.GL_TRIANGLES)


class GSDScene(object):

    def __init__(self, gsd_dict):
        self.lights = create_lights(gsd_dict)
        self.materials = {}
        for material_name in list_material_names(gsd_dict):
            self.materials[material_name] = create_material(gsd_dict,
                                                            material_name)
        self.renderables = []

        print('Loading mesh {}'.format(gsd_dict['mesh']))
        mesh = wavefront.read_obj_file(gsd_dict['mesh'])
        mesh.resize(100)
        for material_id, material_name in enumerate(mesh.materials.keys()):
            filter = {'material': material_id}
            vertex_positions = mesh.expand_face_vertices(filter)
            vertex_normals = mesh.expand_face_normals(filter)
            vertex_tangents, vertex_bitangents = mesh.expand_tangents(
                filter)
            vertex_uvs = mesh.expand_face_uvs(filter)
            material = self.materials[material_name]
            attributes = {
                'a_position': vertex_positions,
                'a_normal': vertex_normals,
            }
            if material.has_texture:
                attributes = {
                    **attributes,
                    'a_tangent': vertex_tangents,
                    'a_bitangent': vertex_bitangents,
                    'a_uv': vertex_uvs,
                }
            self.renderables.append(
                Renderable(material, attributes, len(self.lights)))


def create_lights(gsd_dict):
    lights = []
    for gsd_light in gsd_dict['lights']:
        lights.append(Light(gsd_light['position'],
                            gsd_light['intensity']))

    return lights


def list_material_names(gsd_dict):
    return [m for m in gsd_dict['materials'].keys()]


def create_material(gsd_dict, material_name):
    material_dict = gsd_dict['materials'][material_name]
    if material_dict['type'] == 'svbrdf':
        return SVBRDFMaterial(SVBRDF(material_dict['path']))
    elif material_dict['type'] == 'svbrdf_colortransfer':
        return SVBRDFColorTransferMaterial(SVBRDF(material_dict['path']))
    elif material_dict['type'] == 'phong':
        return PhongMaterial(
            material_dict['diffuse'],
            material_dict['specular'],
            material_dict['shininess'])

