import numpy as np
import os
from others.Animate import Animate
from others.Robot import Robot
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D


def generateRandomInitialState():
    return [np.random.uniform(0, 2), np.random.uniform(0, 2), np.random.uniform(0, 2*np.pi)]


def generateRandomControl():
    return [np.random.uniform(-0.5, 0.5), np.random.uniform(-0.9, 0.9)]


def generateRandomSequence():
    sequence = []
    for _ in range(10):
        ctrl = generateRandomControl()
        for _ in range(20):
            sequence.append(ctrl)
    return sequence


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


def valid(r):
    # NOTE: This function was implemented with the help of ChatGPT
    corners = getCorners(r)

    # Check if any corner is less than 0.2 units away from the boundary
    for corner in corners:
        if corner[0] < 0.2 or corner[0] > 1.8 or corner[1] < 0.2 or corner[1] > 1.8:
            return False

    # If all corners are at least 0.2 units away from the boundary, return True
    return True


def save_controls(controls, map_id, seq_id):
    # Create the controls directory if it does not exist
    if not os.path.exists('controls'):
        os.makedirs('controls')

    controls_array = np.array(controls, dtype=object)

    # Save the file with the naming convention "controls_X_Y.npy"
    filename = f"controls_{map_id}_{seq_id}.npy"
    filepath = os.path.join('controls', filename)
    np.save(filepath, controls_array)


def validSequence(sequence, r):
    for ctrl in sequence:
        r.apply(ctrl[0], ctrl[1])
        r.advance()
        if not valid(r):
            return False
    return True

def generate_and_save_sequence(map_id, seq_id):
    initial_pose = generateRandomInitialState()
    r = Robot(initial_pose[0], initial_pose[1], initial_pose[2])

    if not valid(r):
        return False
    # Start with the initial pose
    controls = [initial_pose]  

    for ctrl in generateRandomSequence():
        r.apply(ctrl[0], ctrl[1])
        r.advance()
        if not valid(r):
            return False
            # Append each control to the list
        controls.append(ctrl)  

    save_controls(controls, map_id, seq_id)
    return True

def load_control_sequences(folder_path):
    control_sequences = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.npy'):
            map_id = int(filename.split('_')[1])  
            if map_id < 5:
                controls = np.load(os.path.join(folder_path, filename), allow_pickle=True)
                control_sequences.append((map_id, controls))
    return control_sequences

def plot_sequences(control_sequences, map_folder):
    num_sequences = len(control_sequences)
    if num_sequences > 10:
        print("Warning: More than 10 control sequences found. Only plotting the first 10.")
        num_sequences = 10
    cols = 5  # Number of columns in the subplot grid
    rows = (num_sequences + cols - 1) // cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(20, 8))
    axes = axes.ravel() if num_sequences > 1 else [axes]

    for i, (_, controls) in enumerate(control_sequences):
        if i >= rows * cols:
            break  # Break if trying to access an index beyond the available axes

        ax = axes[i]
        
        map_id = i // 2  # Integer division to get the map ID
        filename = f"controls_{map_id}_{i % 2}.npy"
        ax.set_title(filename)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)

        # Load and plot landmarks
        landmark_file = os.path.join(map_folder, f"landmark_{map_id}.npy")
        if os.path.exists(landmark_file):
            landmarks = np.load(landmark_file)
            ax.scatter(landmarks[:, 0], landmarks[:, 1], c='black', marker='x', label='Landmarks')

        # Initialize robot with initial pose
        initial_pose = controls[0]
        r = Robot(initial_pose[0], initial_pose[1], initial_pose[2])

        # Plot the initial position
        ax.plot(initial_pose[0], initial_pose[1], 'go', label='Start')  # Start position in green

        # Plot the rest of the positions based on controls
        for ctrl in controls[1:]:
            r.apply(ctrl[0], ctrl[1])
            r.advance()
            ax.plot(r.q[0], r.q[1], 'ro')  # Subsequent positions in red

    # Hide any unused subplots
    for j in range(i + 1, rows * cols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def user_confirmation():
    response = input("Do you want to overwrite the existing control files? (yes/no): ")
    return response.lower() in ["yes", "y"]

def get_rect_corner_from_center(center_x, center_y, angle, length, width):
    # Calculate the bottom-left corner from the center
    corner_x = center_x - (length / 2) * np.cos(angle) - (width / 2) * np.sin(angle)
    corner_y = center_y - (length / 2) * np.sin(angle) + (width / 2) * np.cos(angle)
    return corner_x, corner_y

def update_animation(frame, robot, sequence, path_line, ax, robot_rect, robot_length, robot_width):
    """ Update function for the animation. """
    # Apply control to the robot
    v, phi = sequence[frame]
    robot.apply(v, phi)
    robot.advance()

    # Robot's center coordinates
    center_x = robot.q[0]
    center_y = robot.q[1]

    # Update the path line
    new_x_data = np.append(path_line.get_xdata(), center_x)
    new_y_data = np.append(path_line.get_ydata(), center_y)
    path_line.set_data(new_x_data, new_y_data)

    # Calculate the new bottom-left corner of the rectangle
    corner_x = center_x - (robot_width / 2) * np.cos(robot.q[2]) + (robot_length / 2) * np.sin(robot.q[2])
    corner_y = center_y - (robot_width / 2) * np.sin(robot.q[2]) - (robot_length / 2) * np.cos(robot.q[2])

    # Update the position and angle of the rectangle
    robot_rect.set_xy((corner_x, corner_y))
    robot_rect.angle = np.degrees(robot.q[2])  # Convert angle from radians to degrees

    return path_line, robot_rect



def main():
    num_maps = 5  # Total number of maps (0 to 4)
    num_sequences_per_map = 2  # Total number of sequences per map

    # Generate or load control sequences
    if user_confirmation():
        for map_id in range(num_maps):  # Map IDs are 0 through 4
            for seq_id in range(1, num_sequences_per_map + 1):  # Sequence IDs are 1 and 2
                while not generate_and_save_sequence(map_id, seq_id):
                    pass
    else:
        if not os.path.exists('controls') or not os.listdir('controls'):
            print("No existing control files found. Generating new sequences.")
            for map_id in range(num_maps):  # Iterate over map IDs from 0 to 4
                for seq_id in range(1, num_sequences_per_map + 1):  # Sequence IDs are 1 and 2
                    while not generate_and_save_sequence(map_id, seq_id):
                        pass

    # Load control sequences and plot them
    control_sequences = load_control_sequences('controls')
    plot_sequences(control_sequences, 'maps')

    control_sequences = load_control_sequences('controls')
    num_maps = 5  # Total number of maps (0 to 4)
    num_sequences_per_map = 2  # Total number of sequences per map

    sequence_index = 0  # Index to keep track of the current control sequence
    for map_id in range(num_maps):
        for _ in range(num_sequences_per_map):
            if sequence_index < len(control_sequences):
                _, controls = control_sequences[sequence_index]
                initial_state = controls[0]
                sequence = controls[1:]

                # Load corresponding landmarks
                landmark_file = f"maps/landmark_{map_id}.npy"
                if os.path.exists(landmark_file):
                    landmarks = np.load(landmark_file)
                else:
                    print(f"Landmark file {landmark_file} not found.")
                    sequence_index += 1
                    continue  # Skip animation if landmark file not found

                animate_sequence(initial_state, sequence, landmarks)
                sequence_index += 1
            else:
                break  # Exit loop if there are no more control sequences



def animate_sequence(initial_state, sequence, landmarks):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect('equal')

    # Plot landmarks
    ax.scatter(landmarks[:, 0], landmarks[:, 1], c='black', marker='x', label='Landmarks')

    # Initialize the robot
    robot = Robot(*initial_state)
    robot_length = robot.length
    robot_width = robot.width

    # Initialize the path line
    path_line, = ax.plot([], [], color='gray', linewidth=1)

    # Create and add the robot rectangle to the plot
    robot_rect = Rectangle((0, 0), robot_length, robot_width, fill=False)
    ax.add_patch(robot_rect)

    # Create the animation using FuncAnimation
    ani = FuncAnimation(fig, update_animation, frames=len(sequence), 
                        fargs=(robot, sequence, path_line, ax, robot_rect, robot_length, robot_width), 
                        interval=100, blit=False, repeat=False)

    plt.show()

    

if __name__ == "__main__":
    main()

