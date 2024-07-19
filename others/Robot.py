import numpy as np


def rotate(x, y, theta, x_0, y_0):
    # Uses 2D rotational matrix to rotate (x, y) about (x_0, y_0) by an angle of theta radians
    x -= x_0
    y -= y_0
    c, s = np.cos(theta), np.sin(theta)
    return [c * x - s * y + x_0, s * x + c * y + y_0]


class Robot:

    # This class is used to maintain and modify the state of the robot

    def __init__(self, x=1.0, y=1.0, theta=0.0):
        # IMPORTANT -- cast to float() as there was some kind of numpy error where 1 + 0.01 became 1 but 1.0 + 0.01 resulted in 1.01
        # Reproduce bug with self.q = np.array([1, 1, 0])

        # Initial state [x, y, theta]
        self.q = np.array([float(x), float(y), float(theta)])

        # Control input [v, phi]
        self.u = np.array([0.0, 0.0])

        # Dimensions
        self.length = 0.1
        self.width = 0.2

    def advance(self):
        # Time step
        dt = 0.1

        # Update
        self.q[0] += self.u[0] * np.cos(self.q[2]) * dt
        self.q[1] += self.u[0] * np.sin(self.q[2]) * dt
        # NOTE: It is crucial to update q[2] last as to not improperly update q[0] and q[1] as they depend on q[2]
        self.q[2] += self.u[1] * dt

    def apply(self, v, phi):
        # Control limits
        v_min, v_max = -0.5, 0.5
        phi_min, phi_max = -0.9, 0.9

        # Clip
        self.u[0] = np.clip(v, v_min, v_max)
        self.u[1] = np.clip(phi, phi_min, phi_max)

    def propagate(self, ctrl):
        self.apply(ctrl[0], ctrl[1])
        self.advance()

    def update(this, newQ):
        # directly updates this.q
        this.q[0] = newQ[0]
        this.q[1] = newQ[1]
        this.q[2] = newQ[2]

    def getCorners(self):
        l = self.length
        w = self.width
        q = self.q

        return [
            rotate(q[0] + l/2, q[1] + w/2, q[2], q[0], q[1]),
            rotate(q[0] - l/2, q[1] + w/2, q[2], q[0], q[1]),
            rotate(q[0] + l/2, q[1] - w/2, q[2], q[0], q[1]),
            rotate(q[0] - l/2, q[1] - w/2, q[2], q[0], q[1])
        ]
