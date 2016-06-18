from vispy import app
from svbrdf import io, renderer


svbrdf = io.SVBRDF('/projects/grail/kparnb/Research/Gravel/data/twoshot-svbrdf/leather_black')
canvas = renderer.Canvas(svbrdf, size=(800, 600))

app.run()
