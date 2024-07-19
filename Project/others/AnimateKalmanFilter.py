
import numpy as np
from others.Canvas import Canvas
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from others.KalmanFilter import KalmanFilter
from others.Robot import Robot
from others.DeadReckoning import contruct_landmarks_from_sensor_data
from scipy.optimize import minimize

# NOTE: Trilateration was implemented with the help of ChatGPT


def distance(x, y, landmark):
    return np.sqrt((x - landmark[0])**2 + (y - landmark[1])**2)


def trilateration_objective(coords, *args):
    x, y = coords
    distances, landmarks = args
    return sum((distance(x, y, landmark) - distance_observed)**2 for distance_observed, landmark in zip(distances, landmarks))


def construct_location(landmarks, distances):
    return minimize(trilateration_objective, (0, 0), args=(distances, landmarks))


def calculate_angle(x, y, landmark_x, landmark_y, landmark_angle):
    return np.arctan2(landmark_y - y, landmark_x - x) - landmark_angle


def construct_orientation_angle(x, y, landmarks, landmark_angles):
    sum = 0.0
    index = 0
    for landmark in landmarks:
        sum += calculate_angle(x, y,
                               landmark[0], landmark[1], landmark_angles[index])
        index += 1
    return sum / len(landmark_angles)


def construct_observation(landmarks, sensor_data):
    index = 0
    distances = []
    angles = []
    while index < len(sensor_data):
        distances.append(sensor_data[index])
        angles.append(sensor_data[index + 1])
        index += 2
    loc = construct_location(landmarks, distances)
    ang = construct_orientation_angle(loc.x[0], loc.x[1], landmarks, angles)
    return [loc.x[0], loc.x[1], ang]


def AnimateKalmanFilter(map, measurements, estimates, high_noise=False):
    # true landmark locations
    landmarks = np.load(map, allow_pickle=True)

    # measurement data
    measurements = np.load(measurements, allow_pickle=True)

    # Initialize Canvas
    canvas = Canvas(800, 800, 0, 2, 0, 2)

    # Create a figure and use the ax from the Canvas
    fig = plt.gcf()
    ax = canvas.ax

    # Initialize readings robot
    readings_robot = Robot(measurements[0][0],
                           measurements[0][1], measurements[0][2])
    kalman_robot = Robot(measurements[0][0],
                         measurements[0][1], measurements[0][2])

    # Initial mean and covariance matrix
    mu = np.array(measurements[0])
    Sigma = np.array([[0.075 * np.cos(mu[2]), 0, 0],
                      [0, 0.075 * np.sin(mu[2]), 0],
                      [0, 0, 0.2]])

    def setMu(m):
        global mu
        mu = m

    def setParameters(params):
        global mu
        global Sigma
        mu, Sigma = params
        mu = [params[0][0], params[0][1], params[0][2]]

    # Create an update function for the animation
    measurements = measurements[1:]
    N = len(measurements) // 2

    def update(frame):
        if frame == N - 2:
            ani.event_source.stop()

        # Apply odometry readings and advance readings robot
        r_index = (frame + 1) * 2
        readings_robot.apply(
            measurements[r_index][0], measurements[r_index][1])
        readings_robot.advance()

        # Call the Kalman Filter
        z_t = np.array(construct_observation(
            landmarks, measurements[r_index + 1]))
        setParameters(KalmanFilter(
            mu, Sigma, np.array(measurements[r_index]), z_t))
        kalman_robot.update(mu)
        kalman_robot.propagate(measurements[r_index])
        setMu(kalman_robot.q)

        # Clear the axis and add the updated particles
        ax.clear()
        ax.set_xlim(canvas.x_min, canvas.x_max)
        ax.set_ylim(canvas.y_min, canvas.y_max)

        # Add ground_truth landmarks
        canvas.add_landmarks(landmarks, color='b')

        # Add noisy landmarks
        constructed_landmarks = contruct_landmarks_from_sensor_data(
            readings_robot.q[0], readings_robot.q[1], readings_robot.q[2],
            measurements[r_index+1])
        canvas.add_robot(readings_robot, color='r')
        canvas.add_robot(kalman_robot, color='k')
        canvas.add_landmarks(constructed_landmarks, color='r', marker='x')

    num_frames = N - 1
    ani = FuncAnimation(fig, update, frames=num_frames,
                        interval=1, blit=False)

    # Display the animation
    plt.show()
