import vispy
from vispy import app
from svbrdf import renderer, SVBRDF
import socket

if socket.gethostname() == 'Konstantinoss-MacBook-Pro.local':
    app.use_app('glfw')
    datapath = '/Users/kostas/Documents/Projects/wood_door'
else:
    app.use_app('pyglet')
    datapath = '/Volumes/Seagate/Research/data/svbrdf/twoshot_data_results/leather_antique'

svbrdf = SVBRDF(datapath)
canvas = renderer.Canvas(svbrdf, size=(800, 600))

app.run()
