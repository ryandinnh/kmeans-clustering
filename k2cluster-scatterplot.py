import numpy as np
import matplotlib.pyplot as plt

def initialize_centers(data, k):
    n = data.shape[0]
    indices = np.random.choice(n, k, replace=False)
    centers = data[indices]
    return centers

def kmeans(data, k):
    # Initialize cluster centers
    centers = initialize_centers(data, k)

    while True:
        # Assign each point to the nearest cluster center
        distances = np.sqrt(((data - centers[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        # Update cluster centers
        new_centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        if np.allclose(new_centers, centers):
            break

        centers = new_centers
        
    return labels

data = np.loadtxt('/2D_data.txt')

labels = kmeans(data, k=2)

#making cluster 2, blue
colors = ['red', 'blue']
for i in range(2):
    plt.scatter(data[labels == i, 0], data[labels == i, 1], c=colors[i])

# Show the plot
plt.show()
