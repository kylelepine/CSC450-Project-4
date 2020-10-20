import numpy as np
from cv2 import cv2

cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
fgbg = cv2.createBackgroundSubtractorMOG2()

def distanceCalculation(frame, w, x, y):
    font = cv2.FONT_HERSHEY_COMPLEX
    distance = 0
    fall = False
    cv2.putText(frame, str(w), (20,100), font, 5, (0,255,255), 2, cv2.LINE_AA)

    if(w >= 250):
        distance = 0
        fall = True
        cv2.putText(frame, "FALLEN", (x,y), font, 0.8, (255,0,0), 2, cv2.LINE_AA)
    if(w < 250 and w >= 120):
        distance = 5
        fall = False
        cv2.putText(frame, "0-5 FT", (x,y), font, 0.8, (0,255,255), 2, cv2.LINE_AA)
    if(w < 120 and w >= 100):
        distance = 10
        fall = False
        cv2.putText(frame, "5-10 FT", (x,y), font, 0.8, (0,255,255), 2, cv2.LINE_AA)
    if(w < 100 and w >= 60):
        distance = 15
        fall = False
        cv2.putText(frame, "10-15 FT", (x,y), font, 0.8, (0,255,255), 2, cv2.LINE_AA)
    if(w < 60 and w >= 40):
        distance = 20
        fall = False
        cv2.putText(frame, "15-20 FT", (x,y), font, 0.8, (0,255,255), 2, cv2.LINE_AA)
    if(w < 40 and w >= 20):
        distance = 25
        fall = False
        cv2.putText(frame, "20-25 FT", (x,y), font, 0.8, (0,255,255), 2, cv2.LINE_AA)
    if(w < 20):
        distance = 26
        fall = False
        cv2.putText(frame, "25 FT+", (x,y), font, 0.8, (0,0,0), 2, cv2.LINE_AA)
    if(fall == True):
        print("Ive fallen and I can't get up!")


while(1):
    ret, frame = cap.read()
    frame2 = frame
    fgmask = fgbg.apply(frame)

    blur = cv2.GaussianBlur(fgmask,(5,5),0)
    ret, thresh = cv2.threshold(blur,91,255,cv2.THRESH_BINARY)

    contours =  cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea)
        
        x,y,w,h = cv2.boundingRect(c)
        boxArea = (w-x) * (h-y)
        minArea = 10000

        if(abs(boxArea) > minArea):            
            if(w-x + 40 > h-y):    
                cv2.rectangle(frame2,(x,y),(x+w,y+h),(0,0,255),2)
            else:
                cv2.rectangle(frame2,(x,y),(x+w,y+h),(0,255,0),2)

        distanceCalculation(frame, w, x, y)

    #Show Frames                
    cv2.imshow('frame',frame)
    cv2.imshow('fgmask',fgmask)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    
cap.release()
cv2.destroyAllWindows()
