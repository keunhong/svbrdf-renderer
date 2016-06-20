import numpy as np


def load_pfm_texture(filename: str):
    with open(filename, 'rb') as f:
        header_magic = f.readline().decode().strip()
        header_dims = f.readline().decode().strip()
        _ = f.readline().decode()
        width, height = [int(i) for i in header_dims.split(' ')]
        tex = np.fromfile(f, dtype=np.float32)
        tex = tex.reshape((height, width, 3))

        print('magic={}, width={}, height={}, min={:.2f}, max={:.2f}, mean={:.2f}'.format(
            header_magic, width, height, tex.min(), tex.max(), tex.mean()))

    return tex
