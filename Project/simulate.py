
import argparse
import numpy as np
from others.Animate import Animate
from others.Robot import Robot


def add_actuation_noise(ctrl):
    v_planned = ctrl[0]
    phi_planned = ctrl[1]

    if v_planned == 0:
        v_executed = 0
    else:
        v_executed = np.clip(np.random.normal(v_planned, 0.075), -0.5, 0.5)

    if phi_planned == 0:
        phi_executed = 0
    else:
        phi_executed = np.clip(np.random.normal(phi_planned, 0.2), -0.9, 0.9)

    return [v_executed, phi_executed]


def get_executed_controls(planned_controls):
    executed_controls = []
    # takes in the planned_controls and returns the executed_controls (adds noise)
    for ctrl in planned_controls[1:]:
        executed_controls.append(add_actuation_noise(ctrl))
    return [planned_controls[0]] + executed_controls


def add_odometry_noise(ctrl, high_noise=False):
    v_executed = ctrl[0]
    phi_executed = ctrl[1]

    sigma_ev = 0.05
    sigma_ephi = 0.1

    if high_noise:
        sigma_ev = 0.1
        sigma_ephi = 0.3

    if v_executed == 0:
        v_sensed = 0
    else:
        v_sensed = np.clip(np.random.normal(v_executed, sigma_ev), -0.5, 0.5)

    if phi_executed == 0:
        phi_sensed = 0
    else:
        phi_sensed = np.clip(np.random.normal(
            phi_executed, sigma_ephi), -0.9, 0.9)

    return [v_sensed, phi_sensed]


def get_sensed_controls(executed_controls, high_noise=False):
    sensed_controls = []
    # takes in the executed controls and returns the sensed_controls (contains noise)
    for ctrl in executed_controls[1:]:
        sensed_controls.append(add_odometry_noise(ctrl, high_noise=high_noise))
    return [executed_controls[0]] + sensed_controls


def visualize_controls(planned, executed, mapNum, poses):
    Animate(mapNum, controls=planned, ground_truth=executed, poses=poses)


def getAngle(x_0, y_0, x, y):
    return np.arctan2(y - y_0, x - x_0)


def getDist(x_0, y_0, x, y):
    return np.linalg.norm([x - x_0, y - y_0])


def landmark_sensor(ground_truth_x, ground_truth_y, ground_truth_theta, landmarks):
    landmarks_local = []
    for landmark in landmarks:
        landmarks_local.append([
            getDist(ground_truth_x, ground_truth_y, landmark[0], landmark[1]),
            getAngle(ground_truth_x, ground_truth_y,
                     landmark[0], landmark[1]) + ground_truth_theta
        ])
    return landmarks_local


def add_sensor_noise(data):
    return [
        np.random.normal(data[0], 0.02),
        np.random.normal(data[1], 0.02)
    ]


def interpreted_sensor_data(sensor_data):
    # takes in the sensor_data and returns the interpretation (adds noise)
    interpretation = []
    for data in sensor_data:
        interpretation.append(add_sensor_noise(data))
    return interpretation


def save_ground_truth_and_readings(planned_controls, landmarks, gt_filename, reading_filename, high_noise):
    executed_controls = get_executed_controls(planned_controls)
    sensed_controls = get_sensed_controls(
        executed_controls, high_noise=high_noise)
    robot = Robot(
        executed_controls[0][0], executed_controls[0][1], executed_controls[0][2])

    ground_truth = []
    readings = []

    # VERY IMPORTANT -- don't append robot.q this appends the obejct and not a copy of the array,
    # if robot.q is appeneded then upon subsequence modifications to robot.q ground_truth also changes
    ground_truth.append([robot.q[0], robot.q[1], robot.q[2]])
    readings.append([robot.q[0], robot.q[1], robot.q[2]])

    index = 1
    for ctrl in executed_controls[1:]:
        # execute ctrl and append ground_truth data
        robot.apply(ctrl[0], ctrl[1])
        robot.advance()
        ground_truth.append([robot.q[0], robot.q[1], robot.q[2]])

        # append sensed_controls
        readings.append(sensed_controls[index])

        # append sensed_landmark_data
        landmark_data = []
        for data in interpreted_sensor_data(landmark_sensor(robot.q[0], robot.q[1], robot.q[2], landmarks)):
            landmark_data.append(data[0])
            landmark_data.append(data[1])
        readings.append(landmark_data)

        index += 1

    # Save ground_truth and readings
    np.save(gt_filename, np.array(ground_truth, dtype=object))
    np.save(reading_filename, np.array(readings, dtype=object))

    return ground_truth, executed_controls, readings


def run_simulations_and_save_data():
    # Set a global seed
    np.random.seed(42)

    for x in range(0, 5):
        for y in range(1, 3):
            for z in ['L', 'H']:
                planned_controls = np.load(
                    'controls/controls_'+str(x)+'_'+str(y)+'.npy', allow_pickle=True)
                landmarks = np.load('maps/landmark_'+str(x) +
                                    '.npy', allow_pickle=True)
                gt_filename = 'gts/gt_'+str(x)+'_'+str(y)+'.npy'
                reading_filename = 'readings/readings_' + \
                    str(x)+'_'+str(y)+'_'+str(z)+'.npy'
                high_noise = reading_filename[-5] == 'H'
                save_ground_truth_and_readings(
                    planned_controls, landmarks, gt_filename, reading_filename, high_noise
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=str)
    parser.add_argument("--map", type=str)
    parser.add_argument("--execution", type=str)
    parser.add_argument("--sensing", type=str)
    args = parser.parse_args()

    planned_controls = np.load(args.plan, allow_pickle=True)
    landmarks = np.load(args.map, allow_pickle=True)
    gt_filename = args.execution
    reading_filename = args.sensing
    high_noise = reading_filename[-5] == 'H'
    mapNum = int(args.map[-5])

    ground_truth, executed_controls, readings = save_ground_truth_and_readings(
        planned_controls, landmarks, gt_filename, reading_filename, high_noise
    )

    visualize_controls(planned_controls, executed_controls,
                       mapNum, ground_truth)
