from cv2 import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

while(True):
        
    try:#Error Occurs if Nothing is Found
          
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)
        
        #Set Viewing Region to Include the Whole Frame
        viewRegion = frame[0:1000, 0:1000]

        #Convert BGR to HSV (Hue, Saturation, Value)
        #HSV is Better for Object Detection Compared to BGR
        hsv = cv2.cvtColor(viewRegion, cv2.COLOR_BGR2HSV)
        
        #Create Skin Color Range in HSV
        darker_skin = np.array([0,20,70])
        lighter_skin = np.array([20,255,255])
        
        #Extract Skin Color
        mask = cv2.inRange(hsv, darker_skin, lighter_skin)

        #Fill Dark Spots Within Light Areas
        mask = cv2.dilate(mask,kernel,iterations = 4)
        
        #Put a Gaussian Blur on the Mask
        mask = cv2.GaussianBlur(mask,(5,5),100) 
        
        #Find Contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        #Find Contour of the Skin Selection
        maxContour = max(contours, key = lambda x: cv2.contourArea(x))
        
        #Approximate the Contour
        epsilon = 0.0001 * cv2.arcLength(maxContour, True)
        approx = cv2.approxPolyDP(maxContour, epsilon, True)
        
        #Make Convex Hull around Skin Area
        hull = cv2.convexHull(maxContour)
        
        #Define Area of Hull and Area of Skin Selection
        areaHull = cv2.contourArea(hull)
        areaMaxContour = cv2.contourArea(maxContour)
    
        #Find the Defects in the Convex Hull
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            
            #Draw Lines Around the Largest Area of Exposed Skin
            cv2.line(viewRegion, start, end, [255,255,0], 2)

        #Show All Windows
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)

    except:
        pass
        
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()    
    