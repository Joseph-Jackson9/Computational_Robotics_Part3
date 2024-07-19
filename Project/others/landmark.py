
import numpy as np
import matplotlib.pyplot as plt


def generate(N, i, seed):
    rng = np.random.default_rng(seed=seed)
    landmarks = []
    for _ in range(N):
        landmarks.append([rng.uniform(0, 2), rng.uniform(0, 2)])
    landmarks = np.array(landmarks)
    np.save('maps/landmark_'+str(i)+'.npy', landmarks)
    return landmarks


def visualize(i):
    landmarks = np.load('maps/landmark_'+str(i)+'.npy', allow_pickle=True)

    # Extract x and y coordinates
    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]

    # Plot the landmarks
    plt.scatter(x_coords, y_coords, marker='o', color='red')

    # Add labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Landmarks Visualization ('+str(i)+')')

    # Set axes limits to [0, 2]
    plt.xlim(0, 2)
    plt.ylim(0, 2)

    # Show the plot
    plt.show()


def generateFiles():
    # NOTE: given the following seeds the same exact files will be produced
    generate(5, 1, 41)
    generate(5, 2, 30)
    generate(12, 3, 399)
    generate(12, 4, 19)


if __name__ == '__main__':
    # NOTE: After running, x-ing one visualization out at a time will reveal the next
    # generateFiles()
    for i in range(5):
        visualize(i)
