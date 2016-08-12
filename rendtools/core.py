import numpy as np
from numpy import linalg
from vispy import gloo, app
from . import vector_utils


class Program:
    def __init__(self, vert_shader, frag_shader):
        self._vert_shader = vert_shader
        self._frag_shader = frag_shader

    def compile(self, num_lights):
        vs = self._vert_shader.substitute()
        fs = self._frag_shader.substitute(num_lights=num_lights)
        return gloo.Program(vs, fs)


class Renderable:
    def __init__(self, material, attributes, num_lights):
        self.num_lights = num_lights
        self.material = material
        self.program = material.compile(self.num_lights)
        self.attributes = attributes
        self.update()

    def update(self):
        self.program = self.material.compile(self.num_lights)
        for k, v in self.attributes.items():
            self.program[k] = v



class Light:
    def __init__(self, position, intensity, color=(1.0, 1.0, 1.0)):
        self.position = position
        self.intensity = intensity
        self.color = color


class Renderer(app.Canvas):
    def __init__(self, size, near, far, camera, *args, **kwargs):
        super().__init__(size=size, *args, **kwargs)
        gloo.set_state(depth_test=True)
        gloo.set_viewport(0, 0, *self.size)

        self.program = None
        self.size = size
        self.far = far
        self.near = near

        # Buffer shapes are HxW, not WxH...
        self._rendertex = gloo.Texture2D(shape=(size[1], size[0]) + (4,))
        self._fbo = gloo.FrameBuffer(self._rendertex, gloo.RenderBuffer(
            shape=(size[1], size[0])))

        self.camera = camera
        self.model_mat = np.eye(4)

        self.mesh = None

    def set_program(self, vertex_shader, fragment_shader):
        self.program = gloo.Program(vertex_shader, fragment_shader)

    def update_uniforms(self):
        """
        Call when uniforms have changed.
        """
        self.program['u_view'] = self.camera.view_mat().T
        self.program['u_model'] = self.model_mat
        self.program['u_perspective'] = self.camera.perspective_mat().T

    def draw(self):
        """
        Override and implement drawing logic here. e.g. gloo.clear_color
        """
        raise NotImplementedError

    def render_to_image(self):
        """
        Renders to an image.
        :return: image of rendered scene.
        """
        with self._fbo:
            self.draw()
            pixels = gloo.util.read_pixels(out_type=np.float32,
                                           alpha=False)[:, :, 0]
        return pixels

    def on_resize(self, event):
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        self.camera.size = self.size
        self.draw()

    def on_draw(self, event):
        self.draw()

    def on_key_press(self, event):
        if event.key == 'Escape':
            self.app.quit()

    def on_mouse_move(self, event):
        if event.is_dragging:
            self.camera.handle_mouse(event.last_event.pos, event.pos)
            self.update_uniforms()
            self.update()

    def on_mouse_wheel(self, event):
        cur_dist = linalg.norm(self.camera.position)
        zoom_amount = 5.0 * event.delta[1]
        zoom_dir = vector_utils.normalized(self.camera.position)
        if cur_dist - zoom_amount > 0:
            self.camera.position -= zoom_amount * zoom_dir
            self.update_uniforms()
            self.update()
