import os
import numpy as np
from others.Canvas import Canvas
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from others.ParticleFilter import ParticleFilter, Particle
from others.Robot import Robot
from others.DeadReckoning import contruct_landmarks_from_sensor_data

def AnimateParticleFilterKidnapped(map, measurements, num_particles, estimates, high_noise=False):
    # true landmark locations
    landmarks = np.load(map, allow_pickle=True)

    # Modify estimates file name handling
    if estimates.endswith('.npy'):
        estimates = estimates[:-4]
    file_name = f"{estimates}.npy"

    # measurement data
    measurements = np.load(measurements, allow_pickle=True)
    best_poses = []

    # Initialize particles uniformly
    particles = []
    initial_weight = 1.0 / num_particles

    # Define the boundaries of your environment
    x_min, x_max = 0, 2
    y_min, y_max = 0, 2
    theta_min, theta_max = 0, 2 * np.pi

    for _ in range(num_particles):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        theta = np.random.uniform(theta_min, theta_max)
        particles.append(Particle(Robot(x, y, theta), initial_weight))

    # Initialize Canvas
    canvas = Canvas(800, 800, 0, 2, 0, 2)

    # Create a figure and use the ax from the Canvas
    fig = plt.gcf()
    ax = canvas.ax

    # Initialize readings robot
    readings_robot = Robot(measurements[0][0],
                           measurements[0][1], measurements[0][2])

    # Create an update function for the animation
    measurements = measurements[1:]
    N = len(measurements) // 2
    pf = ParticleFilter(particles, measurements[0], landmarks, measurements[1], high_noise)

    def setParticles(p):
        global particles
        particles = p

    def update(frame):
        if frame == N - 2:
            ani.event_source.stop()

        # Initialize the Particle Filter
        r_index = (frame + 1) * 2
        pf = ParticleFilter(
            particles, measurements[r_index], landmarks, measurements[r_index + 1], high_noise)

        # Apply odometry readings and advance readings robot
        readings_robot.apply(
            measurements[r_index][0], measurements[r_index][1])
        readings_robot.advance()

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
        canvas.add_landmarks(constructed_landmarks, color='r', marker='x')

        # Add particles
        for particle in particles:
            canvas.add_landmark(
                [particle.robot.q[0], particle.robot.q[1]], color='g')

        # Add best estimate
        canvas.add_landmark([pf.best[0], pf.best[1]], color='k')
        best_poses.append(pf.best)
        # Update particles for next iteration
        setParticles(pf.particles)

    
    
    num_frames = N - 1
    ani = FuncAnimation(fig, update, frames=num_frames,
                        interval=1, blit=False)

    # Display the animation
    plt.show()

    def save_best_poses(filename):  
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Save as a 2D numpy array
        np.save(filename, np.array(best_poses))
    
    save_best_poses(file_name)