import os
from . import io

MAP_DIFF_FNAME = 'map_diff.pfm'
MAP_SPEC_FNAME = 'map_spec.pfm'
MAP_NORMAL_FNAME = 'map_normal.pfm'
MAP_SPEC_SHAPE_FNAME = 'map_spec_shape.pfm'
MAP_PARAMS_FNAME = 'map_params.dat'


class SVBRDF:
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError('The path {} does not exist'.format(path))

        path = os.path.join(path, 'out/reverse')

        with open(os.path.join(path, MAP_PARAMS_FNAME), 'r') as f:
            line = f.readline()
            self.alpha, _ = [float(i) for i in line.split(' ')]

        print('Loading diffuse map.')
        self.diffuse_map = io.load_pfm_texture(
            os.path.join(path, MAP_DIFF_FNAME))
        print('Loading specular map.')
        self.specular_map = io.load_pfm_texture(
            os.path.join(path, MAP_SPEC_FNAME))
        print('Loading normal map.')
        self.normal_map = io.load_pfm_texture(
            os.path.join(path, MAP_NORMAL_FNAME))
        print('Loading specular shape map.')
        self.spec_shape_map = io.load_pfm_texture(
            os.path.join(path, MAP_SPEC_SHAPE_FNAME))

        print('Loaded SVBRDF with width={}, height={}, alpha={}'.format(
            self.diffuse_map.shape[0], self.diffuse_map.shape[1], self.alpha))
