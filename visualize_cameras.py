import cv2
import numpy as np
import matplotlib.pyplot as plt

#load rvecs and tvecs

rvecs0 = np.load("calibration/lenovo/rvecs.npy")
tvecs0 = np.load("calibration/lenovo/tvecs.npy")

rvecs2 = np.load("calibration/logitech/rvecs.npy")
tvecs2 = np.load("calibration/logitech/tvecs.npy")


#draw cameras
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#draw camera 0
ax.scatter(tvecs0[0], tvecs0[1], tvecs0[2], c='r', marker='o')

#draw camera 2
ax.scatter(tvecs2[0], tvecs2[1], tvecs2[2], c='b', marker='o')

#visualize camera rotation

#draw camera 0
rmat, _ = cv2.Rodrigues(rvecs0)
arrow_top = tvecs0 + rmat @ np.array([[0, 0, 1]]).T

arrow_top = arrow_top.squeeze()
tvecs0 = tvecs0.squeeze()

ax.plot([tvecs0[0], arrow_top[0]], [tvecs0[1], arrow_top[1]], [tvecs0[2], arrow_top[2]], c='r')

#draw camera 2
rmat, _ = cv2.Rodrigues(rvecs2)
arrow_top = tvecs2 + rmat @ np.array([[0, 0, 1]]).T

arrow_top = arrow_top.squeeze()
tvecs2 = tvecs2.squeeze()

ax.plot([tvecs2[0], arrow_top[0]], [tvecs2[1], arrow_top[1]], [tvecs2[2], arrow_top[2]], c='b')

ax.plot([0, 0], [0, 0], [0, 1], c='g')

plt.show()