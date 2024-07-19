import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D


class Canvas:
    # This class is used as a wrapper to handle all the plotting
    def __init__(self, width, height, x_min, x_max, y_min, y_max, dim=0, resolution=800):
        self.width = width
        self.height = height
        self.fig, self.ax = plt.subplots(figsize=(width / 100, height / 100))
        self.ax.axis('on')
        self.ax.set_aspect('equal')
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

        if dim > 0:
            self.ax.set_xlim(-dim, dim)
            self.ax.set_ylim(-dim, dim)
            self.x_min, self.x_max = -dim, dim
            self.y_min, self.y_max = -dim, dim
        else:
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)

    def draw_rotated_rectangle(self, ax, x, y, width, height, angle_degrees, color='b'):
        rect = patches.Rectangle((x - width / 2, y - height / 2), width,
                                 height, linewidth=1, edgecolor=color, facecolor=color)
        t = Affine2D().rotate_deg_around(x, y, angle_degrees) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

    def add_robot(self, robot, color='b'):
        self.draw_rotated_rectangle(
            self.ax, robot.q[0], robot.q[1], robot.length, robot.width, np.degrees(robot.q[2]), color=color)

    def add_line(self, x1, y1, x2, y2):
        self.ax.plot([x1, x2], [y1, y2], marker='',
                     linestyle='-', color='r', label='Line Segment')

    def add_landmark(self, landmark, R=0.01, color='k', marker='o'):
        if marker == 'o':
            self.ax.add_patch(patches.Circle(
                landmark, R, edgecolor=color, facecolor=color))
        else:
            self.ax.scatter(landmark[0], landmark[1],
                            marker=marker, color=color)

    def add_path(self, pts, color='r'):
        for pt in pts:
            self.add_landmark(pt, color=color)

    def add_landmarks(self, landmarks, color='b', marker='o'):
        for landmark in landmarks:
            self.add_landmark(landmark, color=color, marker=marker)

    def show(self):
        plt.show()
