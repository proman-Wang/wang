import cv2
import numpy as np
import time
import os
import HandCrackingModule as htm

brushThickness = 15
eraserThickness = 50
gametype = 1
gamestart = 1

folderPath2 = "Window"

myList2 = os.listdir(folderPath2)

windowList = []

for imPath1 in myList2:
    image2 = cv2.imread(f'{folderPath2}/{imPath1}')
    windowList.append(image2)

window = windowList[0]
#默认颜色
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    if gamestart == 1:
        img = window
        cv2.imshow("mainwindows", img)
        cv2.waitKey(1)

    #1、导入图片
    success, img = cap.read()
    success, q = cap.read()
    cv2.imshow("11", q)
    cv2.waitKey(1)
    #图片翻转，向左右画画以人的视角
    img = cv2.flip(img, 1)

    cv2.namedWindow("mainwindows", cv2.WINDOW_NORMAL)

    # cv2.imshow("windows", windows)
    # cv2.waitKey(1)

    #2、找到手部坐标
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    # img[0:720, 0:1280] = window
    # windows = detector.findHands(windows)
    # lmList2 = detector.findPosition(windows, draw=False)

    if len(lmList)!= 0 and gametype == 1:
        #print(lmList)

        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        #3、检测手指向上指
        fingers = detector.fingersUp()
        #print(fingers)

        #4、进入选择状态
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if 500 < x1 < 775:
                if 250 < y1 < 375:
                    gametype = 0
                    gamestart = 0
                    window = windowList[3]
                    cv2.destroyWindow("mainwindows")
                if 425 < y1 < 540:
                    window = windowList[2]
            else:
                window = windowList[0]

    if len(lmList) != 0 and gametype == 0:

         x1, y1 = lmList[8][1:]
         x2, y2 = lmList[12][1:]
         # 3、检测手指向上指
         fingers = detector.fingersUp()
         # print(fingers)

         # 4、进入选择状态
         if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            #print("Selection Mode")
            if y1 < 125:
                if 235<x1<325:
                    window = windowList[3]
                    drawColor = (0, 0, 255)
                if 475<x1<575:
                    window = windowList[4]
                    drawColor = (255, 0, 0)
                if 700<x1<775:
                    window = windowList[5]
                    drawColor = (0, 255, 0)
                if 890<x1<1025:
                    window = windowList[6]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

            #5、进入绘画状态
         if fingers[1] and fingers[2]== False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            #print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    if gamestart == 0:
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        img[0:125, 0:1280] = window
        cv2.imshow("mainwindows", img)
        cv2.waitKey(1)



