import cv2
import numpy as np
import os

def find_cameras(open = False):

    cams = []

    for i in range(10):
        vc = cv2.VideoCapture(i)
        if vc.isOpened():
            cams.append(i)
            vc.release()

    return cams

cams = find_cameras()
print(cams)