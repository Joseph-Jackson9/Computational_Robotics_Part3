import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import argparse
import matplotlib.transforms as transforms
from others.Robot import Robot

def angular_difference(angle1, angle2):
    diff = (angle1 - angle2 + np.pi) % (2 * np.pi) - np.pi
    return np.abs(diff)

def getCorners(r):
    center_x = r.q[0]
    center_y = r.q[1]
    length = r.length
    width = r.width
    angle = r.q[2]

    # Calculate half-length and half-width
    half_length = length / 2
    half_width = width / 2

    # Calculate the coordinates of the four corners
    corners = [
        (
            center_x + half_length *
            np.cos(angle) - half_width * np.sin(angle),
            center_y + half_length * np.sin(angle) + half_width * np.cos(angle)
        ),
        (
            center_x - half_length *
            np.cos(angle) - half_width * np.sin(angle),
            center_y - half_length * np.sin(angle) + half_width * np.cos(angle)
        ),
        (
            center_x - half_length *
            np.cos(angle) + half_width * np.sin(angle),
            center_y - half_length * np.sin(angle) - half_width * np.cos(angle)
        ),
        (
            center_x + half_length *
            np.cos(angle) + half_width * np.sin(angle),
            center_y + half_length * np.sin(angle) - half_width * np.cos(angle)
        )
    ]

    return corners

def load_data(file_path):
    if os.path.exists(file_path):
        return np.load(file_path, allow_pickle=True)
    else:
        print(f"File {file_path} not found.")
        return None



def get_rect_corner_from_center(center_x, center_y, angle, length, width):
    # Calculate the bottom-left corner from the center
    corner_x = center_x - (length / 2) * np.cos(angle) - (width / 2) * np.sin(angle)
    corner_y = center_y - (length / 2) * np.sin(angle) + (width / 2) * np.cos(angle)
    return corner_x, corner_y

def animate_sequence(initial_pose_gt, sequence_gt, initial_pose_est, sequence_est, landmarks):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect('equal')

    # Plot ground truth landmarks in blue
    gt_landmarks = ax.scatter(landmarks[:, 0], landmarks[:, 1], c='blue', marker='.', label='Ground Truth Landmarks')

    # Initialize the robots
    robot_gt = Robot(*initial_pose_gt)
    robot_est = Robot(*initial_pose_est)

    # Create rectangles for both robots (gt in blue, est in black)
    rectangle_gt = Rectangle((0, 0), robot_gt.length, robot_gt.width, fill=False, edgecolor='blue', label='Ground Truth Robot')
    rectangle_est = Rectangle((0, 0), robot_est.length, robot_est.width, fill=False, edgecolor='black', label='Estimated Robot')
    ax.add_patch(rectangle_gt)
    ax.add_patch(rectangle_est)

    # Initialize the path lines (gt in blue, est in black)
    path_line_gt, = ax.plot([], [], color='blue', linewidth=1, linestyle='-', marker='', label='Ground Truth Path')
    path_line_est, = ax.plot([], [], color='black', linewidth=1, linestyle='-', marker='', label='Estimated Path')

    # Add legend outside the plot
    ax.legend(handles=[gt_landmarks, rectangle_gt, rectangle_est, path_line_gt, path_line_est], 
              loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout to make room for the legend
    plt.subplots_adjust(right=0.85)

    translational_errors = []
    rotational_errors = []
    # Update function for the animation
    def update_animation(frame, robot_gt, sequence_gt, robot_est, sequence_est, rectangle_gt, rectangle_est, path_line_gt, path_line_est, ax):
        # Update ground truth robot
        gt_pose = sequence_gt[frame]
        if isinstance(gt_pose, np.ndarray) and gt_pose.ndim == 2 and gt_pose.shape[0] == 1:
            gt_pose = gt_pose[0]
        robot_gt.update(gt_pose)

        # Update estimated robot
        est_pose = sequence_est[frame]
        if isinstance(est_pose, np.ndarray) and est_pose.ndim == 2 and est_pose.shape[0] == 1:
            est_pose = est_pose[0]
        robot_est.update(est_pose)

        # Update functions for each robot
        update_robot(robot_gt, rectangle_gt, path_line_gt, ax)
        update_robot(robot_est, rectangle_est, path_line_est, ax)

        # Compute error
        pos_error = np.linalg.norm([sequence_gt[frame][0] - sequence_est[frame][0], 
                                    sequence_gt[frame][1] - sequence_est[frame][1]])
        ori_error = angular_difference(sequence_gt[frame][2], sequence_est[frame][2])
        translational_errors.append(pos_error)
        rotational_errors.append(ori_error)

        return rectangle_gt, rectangle_est, path_line_gt, path_line_est

    def update_robot(robot, rectangle, path_line, ax):
        center_x, center_y, theta = robot.q
        corner_x = center_x - robot.length/2
        corner_y = center_y - robot.width/2
        rectangle.set_xy((corner_x, corner_y))
        t = transforms.Affine2D().rotate_around(center_x, center_y, theta)
        rectangle.set_transform(t + ax.transData)
        path_line.set_data(np.append(path_line.get_xdata(), center_x), np.append(path_line.get_ydata(), center_y))

    ani = FuncAnimation(fig, update_animation, frames=min(len(sequence_gt), len(sequence_est)), 
                        fargs=(robot_gt, sequence_gt, robot_est, sequence_est, rectangle_gt, rectangle_est, path_line_gt, path_line_est, ax), 
                        interval=100, blit=False, repeat=False)

    plt.show()
 # Subplot 1: Translational Error
    plt.subplot(2, 1, 1)
    plt.plot(translational_errors, label='Translational Error', color='blue')
    plt.xlabel('Time Step')
    plt.ylabel('Translational Error')
    plt.title('Translational Error over Time')
    plt.legend()

    # Subplot 2: Rotational Error
    plt.subplot(2, 1, 2)
    plt.plot(rotational_errors, label='Rotational Error', color='orange')
    plt.xlabel('Time Step')
    plt.ylabel('Rotational Error')
    plt.title('Rotational Error over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main(map_file, execution_file, estimates_file):
    landmarks = load_data(map_file)
    ground_truth_poses = load_data(execution_file)
    estimated_poses = load_data(estimates_file)

    if landmarks is None or ground_truth_poses is None or estimated_poses is None:
        print("One or more files could not be loaded. Exiting.")
        return

    # Check and extract the first estimated pose
    if estimated_poses.shape[1] == 3:
        initial_pose_est = estimated_poses[0]
    else:
        print("Invalid format for estimated poses.")
        return

    initial_pose_gt = ground_truth_poses[0]
    sequence_gt = ground_truth_poses[1:]
    sequence_est = estimated_poses[1:]

    animate_sequence(initial_pose_gt, sequence_gt, initial_pose_est, sequence_est, landmarks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, required=True)
    parser.add_argument("--execution", type=str, required=True)
    parser.add_argument("--estimates", type=str, required=True)
    args = parser.parse_args()

    main(args.map, args.execution, args.estimates)