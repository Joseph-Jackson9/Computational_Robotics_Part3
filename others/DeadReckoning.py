import numpy as np
from others.Canvas import Canvas
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from others.Robot import Robot

# This class is a helper to visualize the dead reckoning solution


def contruct_landmarks_from_sensor_data(ground_truth_x, ground_truth_y, ground_truth_theta, sensor_data):
    landmarks = []
    index = 0
    while index < len(sensor_data):
        distance = sensor_data[index]
        angle = sensor_data[index + 1]
        # IMPORTANT!!! angle - ground_truth_theta not +!!!
        deltaX = distance * np.cos(angle - ground_truth_theta)
        deltaY = distance * np.sin(angle - ground_truth_theta)
        landmarks.append([ground_truth_x + deltaX, ground_truth_y + deltaY])
        index += 2

    return landmarks


def getAngle(x_0, y_0, x, y):
    return np.arctan2(y - y_0, x - x_0)


def getDist(x_0, y_0, x, y):
    return np.linalg.norm([x - x_0, y - y_0])


def landmark_sensor(ground_truth_x, ground_truth_y, ground_truth_theta, landmarks):
    landmarks_local = []
    for landmark in landmarks:
        landmarks_local.append([
            getDist(ground_truth_x, ground_truth_y, landmark[0], landmark[1]),
            ((getAngle(ground_truth_x, ground_truth_y,
                       landmark[0], landmark[1]) + ground_truth_theta) % (2 * np.pi)) - np.pi
        ])
    return landmarks_local


class DeadReckoning:

    def __init__(self, map, ground_truth, readings):
        # (a) true landmark locations
        landmarks = np.load(map, allow_pickle=True)

        # (b) ground_truth motion
        ground_truth = np.load(ground_truth, allow_pickle=True)

        # (c) dead reckoning solution given readings
        readings = np.load(readings, allow_pickle=True)

        # Initialize Canvas
        canvas = Canvas(800, 800, 0, 2, 0, 2)

        # Initialize ground_truth robot
        gt_robot = Robot(ground_truth[0][0],
                         ground_truth[0][1], ground_truth[0][2])
        ground_truth = ground_truth[1:]

        # Initialize readings robot
        readings_robot = Robot(readings[0][0],
                               readings[0][1], readings[0][2])
        readings = readings[1:]

        # Create a figure and use the ax from the Canvas
        fig = plt.gcf()
        ax = canvas.ax

        # Create an update function for the animation
        N = len(ground_truth)

        gt_path = []
        readings_path = []

        def update(frame):
            if frame == N - 2:
                ani.event_source.stop()  # Stop the animation after a specific number of frames

            # Update ground_truth robot
            gt_robot.update(ground_truth[frame])

            # I SPENT HOURS ON THIS BUG!!! -- (frame + 1) * 2 not frame * 2
            r_index = (frame + 1) * 2

            # Apply odometry readings and advance readings robot
            readings_robot.apply(readings[r_index][0], readings[r_index][1])
            readings_robot.advance()

            # Clear the axis and add the updated robots
            ax.clear()
            ax.set_xlim(canvas.x_min, canvas.x_max)
            ax.set_ylim(canvas.y_min, canvas.y_max)

            # Add truth
            canvas.add_landmarks(landmarks, color='b')
            canvas.add_path(gt_path, color='b')
            canvas.add_robot(gt_robot, color='b')
            gt_path.append([gt_robot.q[0], gt_robot.q[1]])

            # Construct Landmarks
            constructed_landmarks = contruct_landmarks_from_sensor_data(
                readings_robot.q[0], readings_robot.q[1], readings_robot.q[2],
                readings[r_index+1])

            # Add readings
            canvas.add_landmarks(constructed_landmarks, color='r', marker='x')
            for landmark in constructed_landmarks:
                canvas.add_line(
                    readings_robot.q[0], readings_robot.q[1], landmark[0], landmark[1])

            canvas.add_path(readings_path, color='r')
            canvas.add_robot(readings_robot, color='r')
            readings_path.append([readings_robot.q[0], readings_robot.q[1]])

            return ax

        num_frames = N - 1
        ani = FuncAnimation(fig, update, frames=num_frames,
                            interval=1, blit=False)

        # Display the animation
        plt.show()
