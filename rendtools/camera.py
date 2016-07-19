import numpy as np
from numpy import linalg
from vispy import util
from vispy.util.quaternion import Quaternion
from . import utils


class BaseCamera:
    def __init__(self, fov, size, near, far, position, lookat, up):
        self.fov = fov
        self.size = size
        self.near = near
        self.far = far

        self.position = np.array(position)
        self.lookat = np.array(lookat)
        self.up = utils.normalized(np.array(up))

    def forward(self):
        return utils.normalized(self.lookat - self.position)

    def perspective_mat(self):
        mat = util.transforms.perspective(
            self.fov, self.size[0] / self.size[1], self.near, self.far).T
        return mat

    def view_mat(self):
        forward = self.forward()
        rotation_mat = np.eye(3)
        rotation_mat[0, :] = utils.normalized(np.cross(forward, self.up))
        rotation_mat[2, :] = -forward
        # We recompute the 'up' vector portion of the matrix as the cross
        # product of the forward and sideways vector so that we have an ortho-
        # normal basis.
        rotation_mat[1, :] = np.cross(rotation_mat[2, :], rotation_mat[0, :])

        position = rotation_mat.dot(self.position)

        view_mat = np.eye(4)
        view_mat[:3, :3] = rotation_mat
        view_mat[:3, 3] = -position

        return view_mat


def _get_arcball_vector(x, y, w, h, r=100.0):
    P = np.array((2.0 * x / w - 1.0,
                  -(2.0 * y / h - 1.0),
                  0))
    OP_sq = P[0] ** 2 + P[1] ** 2
    if OP_sq <= 1:
        P[2] = np.sqrt(1 - OP_sq)
    else:
        P = utils.normalized(P)

    return P


class ArcballCamera(BaseCamera):
    def __init__(self, rotate_speed=100.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rotate_speed = rotate_speed
        self.max_speed = np.pi / 2

    def handle_mouse(self, last_pos, cur_pos):
        va = _get_arcball_vector(*cur_pos, *self.size)
        vb = _get_arcball_vector(*last_pos, *self.size)
        angle = min(np.arccos(min(1.0, np.dot(va, vb))) * self.rotate_speed,
                    self.max_speed)
        print(angle)
        axis_in_camera_coord = np.cross(va, vb)

        cam_to_world = self.view_mat()[:3, :3].T
        axis_in_world_coord = cam_to_world.dot(axis_in_camera_coord)

        rotation_quat = Quaternion.create_from_axis_angle(angle,
                                                          *axis_in_world_coord)
        self.position = rotation_quat.rotate_point(self.position)
