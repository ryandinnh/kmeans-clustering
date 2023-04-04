import numpy as np
import matplotlib.pyplot as plt

def initialize_centers(data, k):
    n = data.shape[0]
    indices = np.random.choice(n, k, replace=False)
    centers = data[indices]
    return centers

def kmeans(data, k, num_runs):
    best_labels = None
    best_centers = None
    best_score = np.inf
    scores = np.zeros(num_runs)

    for i in range(num_runs):
        centers = initialize_centers(data, k)

        while True:
            distances = np.sqrt(((data - centers[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            new_centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])

            if np.allclose(new_centers, centers):
                break

            centers = new_centers

        score = np.sum(np.sqrt(((data - centers[labels])**2).sum(axis=1)))
        scores[i] = score

        if score < best_score:
            best_labels = labels
            best_centers = centers
            best_score = score

    return best_labels, best_centers, scores

data = np.loadtxt('/2D_data.txt')

num_runs = 20
labels, centers, scores = kmeans(data, k=3, num_runs=num_runs)

#distance scores for each run
plt.plot(range(num_runs), scores, 'bo')
plt.xlabel('Run')
plt.ylabel('Distance score')
plt.show()

colors = ['red', 'blue', 'green']
for i in range(3):
    plt.scatter(data[labels == i, 0], data[labels == i, 1], c=colors[i])
plt.show()
