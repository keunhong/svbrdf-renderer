import numpy as np
import argparse
from vispy import app
from meshtools import wavefront
from svbrdf import renderer, SVBRDF

app.use_app('pyglet')

parser = argparse.ArgumentParser()
parser.add_argument('--brdf', dest='brdf_path', type=str, required=True)
parser.add_argument('--obj', dest='obj_path', type=str, required=True)

args = parser.parse_args()

np.set_printoptions(suppress=True)

if __name__=='__main__':
    print('Loading mesh {}'.format(args.obj_path))
    mesh = wavefront.read_obj_file(args.obj_path)
    mesh.resize(100)
    print('Mesh bounding size is {}'.format(mesh.bounding_size()))
    print('Loading BRDF {}'.format(args.brdf_path))
    brdf = SVBRDF(args.brdf_path)
    canvas = renderer.Canvas(brdf, mesh, size=(1000, 1000))
    app.run()
