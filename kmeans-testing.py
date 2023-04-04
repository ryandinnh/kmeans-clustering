import numpy as np
import matplotlib.pyplot as plt

def kmeans(X, k, max_iters=100):
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx]

    for i in range(max_iters):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        cluster_assignment = np.argmin(distances, axis=0)

        for j in range(k):
            centroids[j] = X[cluster_assignment == j].mean(axis=0)

    return centroids, cluster_assignment

X_train = np.loadtxt("/trainX.txt", delimiter=',') #delimeter to break values in text file
y_train = np.loadtxt("/trainY.txt")

# Reshape the images 28^2
X_train = X_train.reshape(-1, 28, 28)

k_values = [5, 10, 15]

for k in k_values:
    centroids, cluster_assignment = kmeans(X_train.reshape(-1, 28*28), k)

    plt.figure(figsize=(10, 4))
    for i in range(k):
        plt.subplot(1, k, i+1)
        plt.imshow(centroids[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title(str(i))
    plt.suptitle("KMeans with k=" + str(k))
    plt.show()

    print("Cluster centroids/mean for k =", k)
    print(centroids)
