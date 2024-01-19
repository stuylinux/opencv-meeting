import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while  cap.isOpened():
    _, frame = cap.read()

    # frame = cv2.resize(frame, (300, 300))

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break


cap.release()

cv2.destroyAllWindows()


