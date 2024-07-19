import numpy as np
import matplotlib.pyplot as plt
landmarks_0 = np.load('landmarks_0.npy', allow_pickle=True)


def generate(N, i):
    landmarks = []
    for _ in range(N):
        landmarks.append([np.random.uniform(0, 2), np.random.uniform(0, 2)])
    landmarks = np.array(landmarks)
    np.save('landmark_'+str(i)+'.npy', landmarks)
    return landmarks


def visualize(i):
    landmarks = np.load('landmark_'+str(i)+'.npy', allow_pickle=True)

    # Extract x and y coordinates
    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]

    # Plot the landmarks
    plt.scatter(x_coords, y_coords, marker='o', color='red')

    # Add labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Landmarks Visualization ('+str(i)+')')

    # Show the plot
    plt.show()


def generateFiles():
    # NOTE: Calling generateFiles() again will overwrite the existing landmark files, be careful!!!
    generate(5, 1)
    generate(5, 2)
    generate(12, 3)
    generate(12, 4)


if __name__ == '__main__':
    # NOTE: After running, x-ing one visualization out at a time will reveal the next
    visualize(1)
    visualize(2)
    visualize(3)
    visualize(4)