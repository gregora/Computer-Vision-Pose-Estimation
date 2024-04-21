import cv2
import numpy as np
import json
import time
import matplotlib.pyplot as plt

cv2.namedWindow("logitech")
cv2.namedWindow("lenovo")

with open("setup.json", "r") as file:
    setup = json.load(file)


logitech = cv2.VideoCapture(setup["logitech"])
lenovo = cv2.VideoCapture(setup["lenovo"])

# get calibration data

mtx1 = np.load("calibration/logitech/mtx.npy")
dist1 = np.load("calibration/logitech/dist.npy")
rvecs1 = np.load("calibration/logitech/rvecs.npy")
tvecs1 = np.load("calibration/logitech/tvecs.npy")

mtx2 = np.load("calibration/lenovo/mtx.npy")
dist2 = np.load("calibration/lenovo/dist.npy")
rvecs2 = np.load("calibration/lenovo/rvecs.npy")
tvecs2 = np.load("calibration/lenovo/tvecs.npy")

positions = []

time_start = time.time()

cv2.imshow("logitech", np.zeros((480, 640, 3), np.uint8))
cv2.imshow("lenovo", np.zeros((480, 640, 3), np.uint8))

cv2.moveWindow("logitech", 0, 0)
cv2.moveWindow("lenovo", 640, 0)


while True:

    rval, frame1 = logitech.read()
    rval2, frame2 = lenovo.read()

    # undistort
    frame1 = cv2.undistort(frame1, mtx1, dist1)
    frame2 = cv2.undistort(frame2, mtx2, dist2)

    #detect red circle

    hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    mask1 = cv2.inRange(hsv1, lower_red, upper_red)
    mask2 = cv2.inRange(hsv2, lower_red, upper_red)

    #erode and dilate

    kernel = np.ones((2, 2), np.uint8)
   
    mask1 = cv2.erode(mask1, kernel, iterations=2)
    mask1 = cv2.dilate(mask1, kernel, iterations=2)

    mask2 = cv2.erode(mask2, kernel, iterations=2)
    mask2 = cv2.dilate(mask2, kernel, iterations=2)
    
    # get connected components

    ret = cv2.connectedComponentsWithStats(mask1)
    stats = ret[2]
    centroids = ret[3]
    areas = stats[:, 4]
    sorted_areas = np.argsort(areas)

    detected1 = False
    detected2 = False

    if(len(sorted_areas) >= 2):
        detected1 = True
        largest_area = sorted_areas[-2]
        centroid1 = centroids[largest_area]
        cv2.circle(frame1, (int(centroid1[0]), int(centroid1[1])), 10, (0, 255, 0), 2)

    ret = cv2.connectedComponentsWithStats(mask2)
    stats = ret[2]
    centroids = ret[3]
    areas = stats[:, 4]
    sorted_areas = np.argsort(areas)

    if(len(sorted_areas) >= 2):
        detected2 = True
        largest_area = sorted_areas[-2]
        centroid2 = centroids[largest_area]
        cv2.circle(frame2, (int(centroid2[0]), int(centroid2[1])), 10, (0, 255, 0), 2)

    cv2.imshow("logitech", frame1)
    cv2.imshow("lenovo", frame2)

    # get 3D position of red circle

    if detected1 and detected2:
        # get rotation matrix
        rmat1, _ = cv2.Rodrigues(rvecs1)
        rmat2, _ = cv2.Rodrigues(rvecs2)

        # get translation vector
        tvec1 = tvecs1.squeeze()
        tvec2 = tvecs2.squeeze()

        # projection matrices
        ext1 = np.zeros((3, 4))
        ext2 = np.zeros((3, 4))

        ext1[:, :3] = rmat1
        ext1[:, 3] = tvec1

        ext2[:, :3] = rmat2
        ext2[:, 3] = tvec2

        P1 = mtx1 @ ext1
        P2 = mtx2 @ ext2

        # triangulate
        position4D = cv2.triangulatePoints(P1, P2, centroid1, centroid2)

        position3D = position4D[:3] / position4D[3]
        print("3D position: ", position3D)

        if time.time() - time_start > 5:
            positions.append(position3D)

            if(len(positions) < 2):
                plt.show()
                continue

            positionsnp = np.array(positions)






    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

#save positions
np.save("positions", positions)

cv2.destroyWindow("logitech")
cv2.destroyWindow("lenovo")
logitech.release()
lenovo.release()