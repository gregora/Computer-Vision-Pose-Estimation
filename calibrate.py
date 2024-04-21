import numpy as np
import cv2
import os
import json
import sys

mode = sys.argv[1]

with open("setup.json", "r") as file:
    setup = json.load(file)

if mode == "calibration":
    camera_name = sys.argv[2]

    camera_id = setup[camera_name]

    vc = cv2.VideoCapture(camera_id)

    cv2.namedWindow(camera_name)

    path = "calibration/" + camera_name + "/"

    img = 0

    while True:
        rval, frame = vc.read()

        cv2.imshow(camera_name, frame)
        key = cv2.waitKey(20)

        if key == 32:
            cv2.imwrite(path + "image" + str(img) + ".jpg", frame)
            img += 1

        if key == 27:
            break
    
    vc.release()

    files = os.listdir("calibration/" + camera_name)

    objpoints = []
    imgpoints = []

    for file in files:
        if file.endswith(".jpg"):
            print("Processing " + camera_name + "/" + file)
            img = cv2.imread("calibration/" + camera_name + "/" + file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9, 7), None)

            #show image with corners
            cv2.drawChessboardCorners(img, (9, 7), corners, ret)
            cv2.imshow("Image", img)
            cv2.waitKey(0)

            if ret:
                objpoints.append(np.zeros((9*7, 3), np.float32))
                objpoints[-1][:, :2] = (np.mgrid[0:9, 0:7] * 2.0).T.reshape(-1, 2)
                imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # save calibration matrices
    np.save(path + "mtx", mtx)
    np.save(path + "dist", dist)

elif mode == "position":

    camera_name = sys.argv[2]

    mtx = np.load("calibration/" + camera_name + "/mtx.npy")
    dist = np.load("calibration/" + camera_name + "/dist.npy")

    camera_id = setup[camera_name]

    vc = cv2.VideoCapture(camera_id)

    while True:
        rval, frame = vc.read()

        frame = cv2.undistort(frame, mtx, dist)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        size = (3, 3)
        ret, corners = cv2.findChessboardCorners(gray, size, None)

        key = cv2.waitKey(20)


        if ret:
            cv2.drawChessboardCorners(frame, size, corners, ret)

            if key == 32:
                objpoints = np.zeros((size[0]*size[1], 3), np.float32)
                objpoints[:, :2] = (np.mgrid[0:size[0], 0:size[1]] * 5.0).T.reshape(-1, 2)

                ret, rvecs, tvecs = cv2.solvePnP(objpoints, corners, mtx, dist)

                np.save("calibration/" + camera_name + "/rvecs", rvecs)
                np.save("calibration/" + camera_name + "/tvecs", tvecs)

                print("Position saved")
                print("rvecs: ", rvecs)
                print("tvecs: ", tvecs)



        cv2.imshow(camera_name, frame)

                
        if key == 27:
            break


