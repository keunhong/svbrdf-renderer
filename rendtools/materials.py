import os
from string import Template

import numpy as np
from vispy.gloo import Texture2D

from .core import Program

_package_dir = os.path.dirname(os.path.realpath(__file__))
_shader_dir = os.path.join(_package_dir, 'shaders')


def _load_shader(name):
    path = os.path.join(_shader_dir, name)
    with open(path, 'r') as f:
        return Template(f.read())


class Material:
    def __init__(self, vert_shader, frag_shader, has_texture=False):
        self.program_tmpl = Program(vert_shader, frag_shader)
        self.has_texture = has_texture

    def update_uniforms(self, program):
        raise NotImplementedError

    def compile(self, num_lights):
        program = self.program_tmpl.compile(num_lights)
        program = self.update_uniforms(program)
        return program


class PhongMaterial(Material):
    def __init__(self, diff_color, spec_color, shininess):
        super().__init__(_load_shader('default.vert.glsl'),
                         _load_shader('phong.frag.glsl'),
                         has_texture=False)
        self.diff_color = diff_color
        self.spec_color = spec_color
        self.shininess = shininess

    def update_uniforms(self, program):
        program['u_diff'] = self.diff_color
        program['u_spec'] = self.spec_color
        program['u_shininess'] = self.shininess
        return program


class SVBRDFMaterial(Material):
    def __init__(self, svbrdf):
        super().__init__(_load_shader('default.vert.glsl'),
                         _load_shader('svbrdf.frag.glsl'),
                         has_texture=True)
        self.alpha = svbrdf.alpha
        self.diff_map = Texture2D(svbrdf.diffuse_map,
                                  interpolation='linear',
                                  wrapping='repeat',
                                  internalformat='rgb32f')
        self.spec_map = Texture2D(svbrdf.specular_map,
                                  interpolation='linear',
                                  wrapping='repeat',
                                  internalformat='rgb32f')
        self.spec_shape_map = Texture2D(svbrdf.spec_shape_map,
                                        wrapping='repeat',
                                        internalformat='rgb32f')
        self.normal_map = Texture2D(svbrdf.normal_map,
                                    interpolation='linear',
                                    wrapping='repeat',
                                    internalformat='rgb32f')

    def update_uniforms(self, program):
        program['alpha'] = self.alpha
        program['diff_map'] = self.diff_map
        program['spec_map'] = self.spec_map
        program['spec_shape_map'] = self.spec_shape_map
        program['normal_map'] = self.normal_map
        return program


class SVBRDFColorTransferMaterial(Material):

    def __init__(self, svbrdf):
        super().__init__(_load_shader('default.vert.glsl'),
                         _load_shader('svbrdf_colortransfer.frag.glsl'),
                         has_texture=True)
        from skimage import color
        print('Converting diffuse map to Lab')
        diff_map_lab = np.clip(svbrdf.diffuse_map, 0, 1)
        diff_map_lab = color.rgb2lab(diff_map_lab).astype(dtype=np.float32)
        self.diff_map_mean = diff_map_lab.mean(axis=(0, 1))
        self.diff_map_std = diff_map_lab.std(axis=(0, 1))
        diff_map_lab = (diff_map_lab - self.diff_map_mean) / self.diff_map_std
        self.spec_scale = 1
        self.spec_shape_scale = 1

        self.alpha = svbrdf.alpha
        self.diff_map = Texture2D(diff_map_lab,
                                  interpolation='linear',
                                  wrapping='repeat',
                                  internalformat='rgb32f')
        self.spec_map = Texture2D(svbrdf.specular_map,
                                  interpolation='linear',
                                  wrapping='repeat',
                                  internalformat='rgb32f')
        self.spec_shape_map = Texture2D(svbrdf.spec_shape_map,
                                        wrapping='repeat',
                                        internalformat='rgb32f')
        self.normal_map = Texture2D(svbrdf.normal_map,
                                    interpolation='linear',
                                    wrapping='repeat',
                                    internalformat='rgb32f')

    def update_uniforms(self, program):
        program['alpha'] = self.alpha
        program['diff_map'] = self.diff_map
        program['spec_map'] = self.spec_map
        program['spec_shape_map'] = self.spec_shape_map
        program['normal_map'] = self.normal_map
        program['source_mean'] = self.diff_map_mean
        program['source_std'] = self.diff_map_std
        program['spec_scale'] = self.spec_scale
        program['spec_shape_scale'] = self.spec_shape_scale
        return program
