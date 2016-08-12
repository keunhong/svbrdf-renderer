import os
import numpy as np
from skimage import io as imio
from skimage import color

from . import io

MAP_DIFF_FNAME = 'map_diff.pfm'
MAP_SPEC_FNAME = 'map_spec.pfm'
MAP_NORMAL_FNAME = 'map_normal.pfm'
MAP_SPEC_SHAPE_FNAME = 'map_spec_shape.pfm'
MAP_PARAMS_FNAME = 'map_params.dat'

def transfer_color(source, target):
    source_lab = color.rgb2lab(source)
    target_lab = color.rgb2lab(target)
    source_mean = source_lab.mean(axis=(0,1))
    source_std = source_lab.std(axis=(0,1))
    target_mean = target_lab.mean(axis=(0,1))
    target_std = target_lab.std(axis=(0,1))
    target_new = (target_lab - target_mean) * source_std / target_std + source_mean
    return np.clip(color.lab2rgb(target_new), 0, 1)


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
        # self.specular_map[:, :, :] = self.specular_map.mean()
        print('Loading normal map.')
        self.normal_map = io.load_pfm_texture(
            os.path.join(path, MAP_NORMAL_FNAME))
        print('Loading specular shape map.')
        self.spec_shape_map = io.load_pfm_texture(
            os.path.join(path, MAP_SPEC_SHAPE_FNAME))

        print('Loaded SVBRDF with width={}, height={}, alpha={}'.format(
            self.diffuse_map.shape[0], self.diffuse_map.shape[1], self.alpha))
