import cv2

cv2.namedWindow("Camera 0")
cv2.namedWindow("Camera 2")

vc0 = cv2.VideoCapture(0)
vc2 = cv2.VideoCapture(2)

if vc0.isOpened(): # try to get the first frame
    rval, frame = vc0.read()
    rval2, frame2 = vc2.read()
else:
    rval = False
    rval2 = False

while rval and rval2:
    cv2.imshow("Camera 0", frame)
    cv2.imshow("Camera 2", frame2)

    rval, frame = vc0.read()
    rval2, frame2 = vc2.read()

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("Camera 0")
cv2.destroyWindow("Camera 2")
vc0.release()
vc2.release()