import numpy as np


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
