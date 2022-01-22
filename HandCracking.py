import cv2
import mediapipe as mp
import time

'''开启摄像头0'''
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

'''之前时间'''
pTime = 0
'''当前时间'''
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            #获取手的坐标位置，并给予编号
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

#获取当前时间,改变帧率
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
#在屏幕左上角显示帧率
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 8, 255), 3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)
