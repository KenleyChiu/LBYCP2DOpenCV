from collections import deque

import cv2 as cv
import numpy as np
from PIL import ImageGrab


def nothing(x):
    pass

def screenCapture():
    screen = ImageGrab.grab()
    screen_np = np.array(screen)
    screen_cap = cv.cvtColor(screen_np, cv.COLOR_BGR2RGB)
    return screen_cap

def setMask(out):
    cv.namedWindow("MaskTrackBar")
    cv.createTrackbar("Lower H", "MaskTrackBar", 0, 180, nothing)
    cv.createTrackbar("Lower S", "MaskTrackBar", 0, 255, nothing)
    cv.createTrackbar("Lower V", "MaskTrackBar", 0, 255, nothing)
    cv.createTrackbar("Upper H", "MaskTrackBar", 180, 180, nothing)
    cv.createTrackbar("Upper S", "MaskTrackBar", 255, 255, nothing)
    cv.createTrackbar("Upper V", "MaskTrackBar", 255, 255, nothing)

    while True:
        ret, frame = cap.read(0)
        frame = cv.flip(frame, 1)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # screen_cap=screenCapture()

        lh = cv.getTrackbarPos("Lower H", "MaskTrackBar")
        ls = cv.getTrackbarPos("Lower S", "MaskTrackBar")
        lv = cv.getTrackbarPos("Lower V", "MaskTrackBar")
        uh = cv.getTrackbarPos("Upper H", "MaskTrackBar")
        us = cv.getTrackbarPos("Upper S", "MaskTrackBar")
        uv = cv.getTrackbarPos("Upper V", "MaskTrackBar")

        lowerhsv = np.array([lh, ls, lv])
        upperhsv = np.array([uh, us, uv])

        kernel = np.ones((5, 5), np.uint8)
        mask = cv.inRange(hsv, lowerhsv, upperhsv)
        mask = cv.erode(mask, kernel, iterations=2)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.dilate(mask, kernel, iterations=1)

        cv.imshow("Original", frame)
        cv.imshow("Filter", mask)
        # out.write(screen_cap)
        k = cv.waitKey(1)
        if k == ord("s"):
            break

    cv.destroyAllWindows()
    return lh, ls, lv, uh, us, uv

def setContours(hsv, lower, upper):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.inRange(hsv, lower, upper)
    mask = cv.erode(mask, kernel, iterations=2)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.dilate(mask, kernel, iterations=1)

    cont, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return cont

def findPointer(contours, center, points, pointIndex) :
    if len(contours) > 0:
        cnt = sorted(contours, key=cv.contourArea, reverse=True)[0]
        # Get the radius of the enclosing circle around the found contour
        ((x, y), radius) = cv.minEnclosingCircle(cnt)
        # Draw the circle around the contour
        cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        # Get the moments to calculate the center of the contour (in this case Circle)
        M = cv.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        points[pointIndex].appendleft(center)
    else:
        points.append(deque(maxlen=512))
        pointIndex += 1
    return points

def drawInk(points, frame, paintWindow, color):
    for i in range(len(points)):
        for j in range(1, len(points[i])):
            if points[i][j - 1] is None or points[i][j] is None:
                continue
            cv.line(frame, points[i][j - 1], points[i][j], color, 10, 0)
            cv.line(paintWindow, points[i][j - 1], points[i][j], color , 10, 0)

def setInkColor():
    color = np.zeros((300, 512, 3), np.uint8)
    cv.namedWindow('Ink Color')

    cv.createTrackbar('B', 'Ink Color', 0, 255, nothing)
    cv.createTrackbar('G', 'Ink Color', 0, 255, nothing)
    cv.createTrackbar('R', 'Ink Color', 0, 255, nothing)

    while True:
        screen_cap = screenCapture()
        cv.imshow('Ink Color', color )
        k = cv.waitKey(1)

        if k == ord('s'):
            cv.destroyWindow('Ink Color')
            return (b,g,r)

        b = cv.getTrackbarPos('B', 'Ink Color')
        g = cv.getTrackbarPos('G', 'Ink Color')
        r = cv.getTrackbarPos('R', 'Ink Color')

        color[:] = [b, g, r]
        out.write(screen_cap)



filename= input("Input the filename of the video: ")

points = [deque(maxlen=512)]
pointIndex = 0

isDrawing = True
color = (255,0,0)



paintWindow = cv.imread('whiteBg.jpg',-1)
dim=(640,480)
paintWindow= cv.resize(paintWindow,dim)


cap = cv.VideoCapture(0)
ret = cap.set(3, 640)
ret = cap.set(4, 480)



fourcc = cv.VideoWriter_fourcc(*'XVID')
out= cv.VideoWriter(filename+'.avi', fourcc, 15, (1920,1080))

#paintWindow = np.zeros((480,640,3))+255

lh, ls, lv, uh, us, uv = setMask(out)

#lowerHSV = np.array([100, 100, 100])
#upperHSV = np.array([140, 255, 255])

lowerHSV = np.array([lh, ls, lv])
upperHSV = np.array([uh, us, uv])


while cap.isOpened():
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    screen_cap=screenCapture()

    if not ret:
        break
    out.write(frame)
    keys= cv.waitKey(1)

    # D to draw
    if keys  == ord('d'):
        isDrawing = True

    # A to stop
    elif keys  == ord('a'):
        isDrawing = False
        points.append(deque(maxlen=512))
        pointIndex += 1

    # C to clear
    elif keys  == ord('c'):
        points = [deque(maxlen=512)]
        pointIndex = 0
        paintWindow[:]= 255

    # F to change Ink Color
    elif keys == ord('f'):
        color = setInkColor()

    elif keys == 27:
        break

    if isDrawing:
        contours = setContours(hsv, lowerHSV, upperHSV)
        center = None
        points = findPointer(contours, center, points, pointIndex)

    drawInk(points, frame, paintWindow, color)

    cv.imshow("Tracking", frame)
    cv.imshow("Paint", paintWindow)
    out.write(screen_cap)

out.release()
cap.release()
cv.destroyAllWindows()
