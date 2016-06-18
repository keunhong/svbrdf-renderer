import os
import numpy as np


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

        self.diffuse_map = load_pfm_texture(
            os.path.join(path, MAP_DIFF_FNAME))
        self.specular_map = load_pfm_texture(
            os.path.join(path, MAP_SPEC_FNAME))
        self.normal_map = load_pfm_texture(
            os.path.join(path, MAP_NORMAL_FNAME))
        self.spec_shape_map = load_pfm_texture(
            os.path.join(path, MAP_SPEC_SHAPE_FNAME))

        print('Loaded SVBRDF with width={}, height={}, alpha={}'.format(
            self.diffuse_map.shape[0], self.diffuse_map.shape[1], self.alpha))


def load_pfm_texture(filename: str):
    with open(filename, 'rb') as f:
        header_magic = f.readline().decode().strip()
        header_dims = f.readline().decode().strip()
        _ = f.readline().decode()
        width, height = [int(i) for i in header_dims.split(' ')]
        print('magic={}, width={}, height={}'.format(
            header_magic, width, height))

        tex = np.fromfile(f, dtype=np.float32)
        tex = tex.reshape((height, width, 3))
    return tex
