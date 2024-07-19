import numpy as np
from simulate import add_odometry_noise, getDist, getAngle
import os


def angular_distance(angle1, angle2):
    if angle2 > angle1:
        temp = angle1
        angle1 = angle2
        angle2 = temp
    return min(abs(angle1 - angle2), abs(angle1 - angle2 - 2 * np.pi))


def likelihood(measurement_dist, measurement_angle, landmark_x, landmark_y, sample):
    # Returns likelihood of an individual measurement
    expected_dist = getDist(
        sample.robot.q[0], sample.robot.q[1], landmark_x, landmark_y)
    expected_angle = ((getAngle(sample.robot.q[0], sample.robot.q[1],
                                landmark_x, landmark_y) + sample.robot.q[2]) % (2 * np.pi))

    diff_distance = abs(expected_dist - measurement_dist)
    diff_angle = angular_distance(expected_angle, measurement_angle)

    # Map difference true and expected distance measurement to probability
    p_z_given_x_distance = \
        np.exp(-(diff_distance) * (diff_distance) /
               (2 * 0.02 * 0.02))

    # Map difference true and expected angle measurement to probability
    p_z_given_x_angle = \
        np.exp(-(diff_angle) * (diff_angle) /
               (2 * 0.02 * 0.02))
    return p_z_given_x_distance * p_z_given_x_angle


class Particle:
    def __init__(self, robot, weight):
        self.robot = robot
        self.weight = weight
        

    def propagate(self, ctrl):
        self.robot.propagate(ctrl)


class ParticleFilter:
    def __init__(self, particles, sensed_control, landmarks, sensor_data, high_noise):
        self.particles = particles  # particles 1 to N for iteration i - 1
        self.best_poses = []
        self.resample(particles)
        self.propagate(sensed_control, high_noise)
        self.reweight(landmarks, sensor_data)
        

    def resample(self, particles):
        weights = np.array([p.weight for p in particles])
        weights += 1.e-30
        normalized_weights = weights / np.sum(weights)
        self.best = self.determine_best(normalized_weights)
        indices = np.random.choice(
            len(particles), len(particles), p=normalized_weights)
        self.particles = [particles[i] for i in indices]

    def determine_best(self, weights):
        best_x = 0
        best_y = 0
        best_theta = 0
        for i in range(len(self.particles)):
            best_x += weights[i] * self.particles[i].robot.q[0]
            best_y += weights[i] * self.particles[i].robot.q[1]
            best_theta += weights[i] * self.particles[i].robot.q[2]
        
        # Ensure best_pose is a flat list or 1D array
        best_pose = [best_x, best_y, best_theta]
        
        # Append the best pose to self.best_poses
        self.best_poses.append(best_pose)
        return (best_x, best_y, best_theta)

    

    def propagate(self, sensed_control, high_noise):
        for p in self.particles:
            p.propagate(add_odometry_noise(
                sensed_control, high_noise=high_noise))

    def reweight(self, landmarks, measurements):
        for p in self.particles:
            # likelihood_measurements = product of the likelihood of each individual measurement
            likelihood_measurements = 1.0

            index = 0
            while index < len(landmarks):
                likelihood_measurements *= likelihood(
                    measurements[2 * index], measurements[2 * index + 1], landmarks[index][0], landmarks[index][1], p)
                index += 1

            p.weight = likelihood_measurements
    
