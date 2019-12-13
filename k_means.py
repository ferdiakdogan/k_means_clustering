import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from centroids import Centroid

# Read input mat file using scipy
x = loadmat('data2.mat')
x = x['X']
plt.figure()
plt.scatter(x[:, 0], x[:, 1], s=50, c='b')
# Set cluster number and initialize them
K = 3
initial_centroids = np.array([[3.0, 3.0], [6.0, 2.0], [8.0, 5.0]])
all_centroids = np.array([initial_centroids])
converged = False


# Find closest centroids
def closest_centroids(x, initial_centroids, all_centroids, converged):
    centroids = []
    for point in x:
        distances = []
        for centroid in initial_centroids:
            distances.append((point[0] - centroid[0])**2 + (point[1] - centroid[1])**2)
        centroids.append(distances.index(min(distances)))
        if distances.index(min(distances)) == 0:
            my_color = 'red'
        elif distances.index(min(distances)) == 1:
            my_color = 'green'
        else:
            my_color = 'blue'
        plt.scatter(point[0], point[1], s=50, c=my_color)
    for i in range(len(all_centroids)):
        if i < len(all_centroids) - 1:
            my_size = 50
        else:
            my_size = 150
        plt.scatter(all_centroids[i, :, 0], all_centroids[i, :, 1], s=my_size, marker='X', c='black')
        if i > 0:
            plt.plot([all_centroids[i, :, 0], all_centroids[i - 1, :, 0]],
                     [all_centroids[i, :, 1], all_centroids[i - 1, :, 1]], c='black')
            if np.array_equal(all_centroids[i, :, :], all_centroids[i - 1, :, :]):
                converged = True
    plt.show()
    return centroids, converged


def compute_centroids(x, initial_centroids, centroids):
    # Find new centers
    for j in range(len(initial_centroids)):
        items = [x[k] for k in range(len(x)) if centroids[k] == j]
        items = sum(items)/len(items)
        initial_centroids[j] = items
    return initial_centroids


while not converged:
    centroids, converged = closest_centroids(x, initial_centroids, all_centroids, converged)
    new_centroids = compute_centroids(x, initial_centroids, centroids)
    all_centroids = np.append(all_centroids, [new_centroids], axis=0)
