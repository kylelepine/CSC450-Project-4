import numpy as np
from cv2 import cv2

# Capture Video Camera
cap = cv2.VideoCapture(cv2.CAP_DSHOW)

#Default Background Subtractor to Detect Movement.
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, detectShadows=False)

#Initialize Default Values
fallFileCount = 0
fiveFileCount = 0
tenFileCount = 0
fifteenFileCount = 0
twentyFileCount = 0
twentyfiveFileCount = 0
startupBuffer = 0

#Set Starting Distance of Subject to 0
distance = 0

#Set Destination Folder to Blank
folder = ""

#Calculate Approxamate Distance to Subject
def distanceCalculation(frame, w, x, y):
    font = cv2.FONT_HERSHEY_COMPLEX
    fall = False

    #Find Distance by Subject's Width Relative to Camera
    if(w >= 250):
        distance = 0
        folder = "FALL"
        fall = True
        cv2.putText(frame, "FALLEN", (x,y), font, 0.8, (255,0,0), 2, cv2.LINE_AA)
    if(w < 250 and w >= 120):
        distance = 5
        folder = "FIVE"
        fall = False
        cv2.putText(frame, "0-5 FT", (x,y), font, 0.8, (0,255,255), 2, cv2.LINE_AA)
    if(w < 120 and w >= 100):
        distance = 10
        folder = "TEN"
        fall = False
        cv2.putText(frame, "5-10 FT", (x,y), font, 0.8, (0,255,255), 2, cv2.LINE_AA)
    if(w < 100 and w >= 60):
        distance = 15
        folder = "FIFTEEN"
        fall = False
        cv2.putText(frame, "10-15 FT", (x,y), font, 0.8, (0,255,255), 2, cv2.LINE_AA)
    if(w < 60 and w >= 40):
        distance = 20
        folder = "TWENTY"
        fall = False
        cv2.putText(frame, "15-20 FT", (x,y), font, 0.8, (0,255,255), 2, cv2.LINE_AA)
    if(w < 40 and w >= 20):
        distance = 25
        folder = "TWENTYFIVE"
        fall = False
        cv2.putText(frame, "20-25 FT", (x,y), font, 0.8, (0,255,255), 2, cv2.LINE_AA)
    if(w < 20):
        distance = 30
        folder = "None"
        fall = False
        cv2.putText(frame, "25 FT+", (x,y), font, 0.8, (0,0,0), 2, cv2.LINE_AA)
    if(fall == True):
        print("Ive fallen and I can't get up!")


#Runs Until Program is Manually Ended.
while(1):
    #Capturing the Frame and Boolean 
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    #Applying Gaussian Blur to fgbg Mask Frame
    blur = cv2.GaussianBlur(fgmask,(5,5),0)

    #Converting Frame with Threshhold (https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html)
    ret, thresh = cv2.threshold(fgmask,91,255,cv2.THRESH_BINARY)
    contours =  cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2] 

    #Buffer Startup
    startupBuffer += 1   

    if (startupBuffer > 200):

        #Find Contours
        if len(contours) != 0:
            c = max(contours, key = cv2.contourArea)
            
            x,y,w,h = cv2.boundingRect(c)
            
            boxArea = w * h
            minArea = 10000

            #Invoke Distance Calculation
            distanceCalculation(frame, w, x, y)

            #Create Template Based on Distance
            if (boxArea > minArea):
                if(distance == 0):
                    crop_frame = fgmask[y:(y+h), x:(x+w)]
                if(distance == 5):
                    crop_frame = fgmask[y:(y+h), x:(x+250)]
                if(distance == 10):
                    crop_frame = fgmask[y:(y+h), x:(x+120)]
                if(distance == 15):
                    crop_frame = fgmask[y:(y+h), x:(x+100)]
                if(distance == 20):
                    crop_frame = fgmask[y:(y+h), x:(x+60)]
                if(distance == 25):
                    crop_frame = fgmask[y:(y+h), x:(x+40)]   
                if(distance == 30):
                    crop_frame = fgmask[y:(y+h), x:(x+20)]
                
                #Create Kernel Sizes
                kernel_close = np.ones((15,15),np.uint8)
                kernel_open = np.ones((1,1),np.uint8)

                #Set Foreground Morph
                foreground_morph = cv2.morphologyEx(crop_frame, cv2.MORPH_OPEN, kernel_open)
                foreground_morph = cv2.morphologyEx(foreground_morph, cv2.MORPH_CLOSE, kernel_close)

                #Assign Templates into Separate Folders by Distance
                if(folder == "FALL"):
                    cv2.imwrite('./templates/morph/test_template/FALL' + str(fallFileCount) + '.png', foreground_morph)
                    fallFileCount += 1
                if(folder == "FIVE"):
                    cv2.imwrite('./templates/morph/test_template/FIVE' + str(fiveFileCount) + '.png', foreground_morph)
                    fiveFileCount += 1
                if(folder == "TEN"):
                    cv2.imwrite('./templates/morph/test_template/TEN' + str(tenFileCount) + '.png', foreground_morph)
                    tenFileCount += 1
                if(folder == "FIFTEEN"):
                    cv2.imwrite('./templates/morph/test_template/FIFTEEN' + str(fifteenFileCount) + '.png', foreground_morph)
                    fifteenFileCount += 1
                if(folder == "TWENTY"):
                    cv2.imwrite('./templates/morph/test_template/TWENTY' + str(twentyFileCount) + '.png', foreground_morph)
                    twentyFileCount += 1
                if(folder == "TWENTYFIVE"):
                    cv2.imwrite('./templates/morph/test_template/TWENTYFIVE' + str(twentyfiveFileCount) + '.png', foreground_morph)
                    twentyfiveFileCount += 1

                cv2.imshow('morphology', foreground_morph)
            
            #Draw Bounding Boxes
            #if(abs(boxArea) > minArea):            
            if(w-x + 40 > h-y):  
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            else:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    #Show Frames            
    cv2.imshow('frame',frame)
    cv2.imshow('fgmask',fgmask)
    
    #Terminate Program with ESC
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
#Terminate Windows
cap.release()
cv2.destroyAllWindows()