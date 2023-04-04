import numpy as np
import matplotlib.pyplot as plt

def initialize_centers(data, k):
    n = data.shape[0]
    indices = np.random.choice(n, k, replace=False)
    centers = data[indices]
    return centers

def kmeans(data, k):
    centers = initialize_centers(data, k)

    while True:
        distances = np.sqrt(((data - centers[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        new_centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        if np.allclose(new_centers, centers):
            break

        centers = new_centers

    score = np.sum(np.sqrt(((data - centers[labels])**2).sum(axis=1)))

    return labels, centers, score

data = np.loadtxt('/2D_data.txt')

ks = [4, 6, 8, 10, 15, 20]

# Perform k-means clustering for each value of K and store the distance scores
scores = []
for k in ks:
    labels, centers, score = kmeans(data, k)
    scores.append(score)

plt.plot(ks, scores, 'bo-')
plt.xlabel('K')
plt.ylabel('Distance score')
plt.show()
