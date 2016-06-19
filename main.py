import argparse
from vispy import app
from svbrdf import renderer, SVBRDF

app.use_app('glfw')

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', dest='input_dir', type=str, required=True)

args = parser.parse_args()


if __name__=='__main__':
    brdf = SVBRDF(args.input_dir)
    canvas = renderer.Canvas(brdf, size=(1280, 800))
    app.run()
