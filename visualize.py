import numpy as np
import matplotlib.pyplot as plt

#load positions

positions = np.load("positions.npy")

positions = positions[:-20]

#plot positions

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2])

plt.show()
