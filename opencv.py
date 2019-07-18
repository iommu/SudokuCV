import cv2, imutils
import numpy as np

img = cv2.imread('test.jpg')
image = imutils.resize(img, width=1000)

img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(img, 127,255,cv2.THRESH_BINARY_INV)
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

large = 0
index = 0


if len(contours):
    c = max(contours,key=cv2.contourArea)
    epsilon = 0.1*cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    cv2.drawContours(image,approx,-1,(0,255,0),3) # gets the four corners of the grid
    outputPoints = np.float32([[0,0,],[0,270],[270,270],[270,0]])
    # sort approx corners
    # Taken from "Scanning documents from photos using OpenCV" by BRETA
    approx = approx[:,0]
    diff= np.diff(approx, axis=1)
    summ = approx.sum(axis=1)
    approx = np.float32([approx[np.argmin(summ)],
    approx[np.argmax(diff)],
    approx[np.argmax(summ)],
    approx[np.argmin(diff)]])
    M = cv2.getPerspectiveTransform(approx,outputPoints)
    image = cv2.warpPerspective(image, M, (270,270))
    cv2.imshow('output', image)
    cv2.waitKey(0)
