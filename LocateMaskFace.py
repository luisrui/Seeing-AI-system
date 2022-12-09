import sys
import numpy as np
import cv2 as cv

modelFile = "appendix/opencv_face_detector_uint8.pb"
configFile = "appendix/opencv_face_detector.pbtxt"
net = cv.dnn.readNetFromTensorflow(modelFile, configFile)
conf_threshold = 0.7


def CropMaskFace(frame, net=net):
    blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), [
                                104, 117, 123], False, False)
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            x1 = x1 if x1 > 0 else 0
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            y1 = y1 if y1 > 0 else 0
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            x2 = x2 if x2 < frameWidth else frameWidth
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            y2 = y2 if y2 < frameHeight else frameHeight
            face_crop = frame[y1:y2, x1:x2]
            # cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # use rectangular to generate the location of faces
            return face_crop, True
    return frame, False
