from vispy import app
from svbrdf import renderer, SVBRDF


svbrdf = SVBRDF('/projects/grail/kparnb/Research/Gravel/data/twoshot-svbrdf/wood_laminate')
canvas = renderer.Canvas(svbrdf, size=(800, 600))

app.run()
