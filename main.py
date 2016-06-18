import vispy
from vispy import app
from svbrdf import renderer, SVBRDF


app.use_app('pyglet')

svbrdf = SVBRDF('/Volumes/Seagate/Research/data/svbrdf/twoshot_data_results/leather_antique')
canvas = renderer.Canvas(svbrdf, size=(800, 600))

app.run()
