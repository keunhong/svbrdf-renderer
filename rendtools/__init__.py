from . import utils
from .camera import *


class Light:
    def __init__(self, position, intensity, color=(1.0, 1.0, 1.0)):
        self.position = position
        self.intensity = intensity
        self.color = color


