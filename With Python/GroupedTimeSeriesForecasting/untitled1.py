# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 15:11:43 2021

@author: Brijesh Rana
"""

import numpy as np
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect objects and draw on screen
    bbox, label, conf = cv.detect_common_objects(frame)
    output_image = draw_bbox(im, bbox, label, conf)

    cv2.imshow('output',output_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()