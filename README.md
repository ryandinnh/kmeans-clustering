# kmeans-clustering
Python implementation of Kmeans clustering for unsupervised learning. Features random initialization of centers, testing on 2D data (K=2 and K=3), and clustering of the MNIST dataset with means for K=5, 10, and 15. Includes code documentation and visualizations.

2D_data.txt is an X and Y value data set used for unclustered and clustered K data plots.

testX.txt, testY.txt, trainX.txt, and trainY.txt are data sets for Kmeans clustering testing.

unclustered-dataplot.py is a scatterplot representation of "2D_data.txt" without K clusters. Cam be modified to represent any data set with X and Y value columns.

k2cluster-scatterplot.py implements Kmeans clustering with K=2 on 2D data using the NumPy and Matplotlib libraries. It contains two functions, "initialize_centers" for randomly initializing the cluster centers and "kmeans" for performing the clustering. The resulting clusters are plotted as a scatterplot using different colors to distinguish between the two clusters.

k3-scatterplot.py has the same implementation as "k2cluster-scatterplot.py" except for with kmeans clustering where k=3. The resulting distance scores are plotted to visualize the performance of the clustering algorithm, along with a scatter plot showing the clustered data points with different colors.

kmulti-distance.py uses the K-means clustering algorithm to perform unsupervised learning on a given 2D dataset, where K ranges from 4 to 20. The script calculates the distance scores for each run, and displays them on a plot to help determine the optimal number of clusters for the given dataset.

