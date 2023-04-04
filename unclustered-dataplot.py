import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('/2D_data.txt')

# create a cluster plot of the data
plt.scatter(data[:,0], data[:,1])
plt.show()
