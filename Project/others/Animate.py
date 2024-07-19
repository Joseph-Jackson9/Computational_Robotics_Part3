import numpy as np
from others.Canvas import Canvas
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from others.Robot import Robot

# This class takes in a map and a set of control inputs and visualizes the corresponding animation


class Animate:

    def __init__(self, mapNum, controls, show_robot=False, skip_to_end=True, title=None, ground_truth=None, animate_ground_truth=False, poses=None):
        # Initialize Canvas
        canvas = Canvas(800, 800, 0, 2, 0, 2)

        landmarks = np.load('maps/landmark_'+str(mapNum) +
                            '.npy', allow_pickle=True)

        canvas.add_landmarks(landmarks)
        robot = Robot(controls[0][0], controls[0][1], controls[0][2])

        if animate_ground_truth:
            robot = Robot(ground_truth[0][0],
                          ground_truth[0][1], ground_truth[0][2])
            ground_truth = ground_truth[1:]

        controls = controls[1:]
        if show_robot:
            canvas.add_robot(robot)

        # Create a figure and use the ax from the Canvas
        fig = plt.gcf()
        ax = canvas.ax

        # Create an update function for the animation
        N = len(controls)

        path = []

        def update(frame):
            if frame == N - 2:
                ani.event_source.stop()  # Stop the animation after a specific number of frames

            if animate_ground_truth:
                robot.apply(ground_truth[frame][0], ground_truth[frame][1])
                col = 'b'
            else:
                robot.apply(controls[frame][0], controls[frame][1])
                col = 'r'
            robot.advance()

            # Clear the axis and add the updated robot
            ax.clear()

            # Add title to the axis
            if title is not None:
                ax.set_title(title)

            ax.set_xlim(canvas.x_min, canvas.x_max)
            ax.set_ylim(canvas.y_min, canvas.y_max)
            canvas.add_landmarks(landmarks)
            canvas.add_path(path, color=col)
            if show_robot:
                canvas.add_robot(robot)
            path.append([robot.q[0], robot.q[1]])

            return ax

        # Create a FuncAnimation object
        if skip_to_end:
            path = []
            path.append([robot.q[0], robot.q[1]])
            for ctrl in controls:
                robot.apply(ctrl[0], ctrl[1])
                robot.advance()
                path.append([robot.q[0], robot.q[1]])
            canvas.add_path(path, color='r')

            if ground_truth is not None:  # executed_path
                # Reset Robot
                robot = Robot(
                    ground_truth[0][0], ground_truth[0][1], ground_truth[0][2])
                ground_truth = ground_truth[1:]

                path = []
                path.append([robot.q[0], robot.q[1]])
                for ctrl in ground_truth:
                    robot.apply(ctrl[0], ctrl[1])
                    robot.advance()
                    path.append([robot.q[0], robot.q[1]])
                canvas.add_path(path, color='b')

            if poses is not None:  # ground_truth poses
                # Reset Robot
                robot = Robot(
                    poses[0][0], poses[0][1], poses[0][2])
                poses = poses[1:]
                path = []
                path.append([robot.q[0], robot.q[1]])
                for pose in poses:
                    robot.update(pose)
                    path.append([robot.q[0], robot.q[1]])
                # canvas.add_path(path, color='g')

        else:
            num_frames = N - 1
            ani = FuncAnimation(fig, update, frames=num_frames,
                                interval=1, blit=False)

        # Add title to the axis
        if title is not None:
            ax.set_title(title)

        # Display the animation
        plt.show()
