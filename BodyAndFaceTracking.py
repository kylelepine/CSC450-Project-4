from __future__ import print_function
from cv2 import cv2 as cv
import argparse

def detectAndDisplay(frame):

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    #-- Detect body
    body = body_cascade.detectMultiScale(frame_gray,1.1,1)
    for (x,y,w,h) in body:
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        frame = cv.resize(frame, (640,480))
    
    #-- Detect profile face (from the side)
    profile = profile_cascade.detectMultiScale(frame_gray)
    for(x,y,w,h) in profile:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)

    #-- Detect face
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x2,y2,w2,h2) in faces:
        center = (x2 + w2//2, y2 + h2//2)
        frame = cv.ellipse(frame, center, (w2//2, h2//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y2:y2+h2,x2:x2+w2]

        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x3,y3,w3,h3) in eyes:
            eye_center = (x2 + x3 + w3//2, y2 + y3 + h3//2)
            radius = int(round((w3 + h3)*0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)

    cv.imshow('Capture - Body And Face Detection', frame)
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--body_cascade', help='Path to body cascade.', default='C:/Users/Logan Koch/Github/haarcascade_fullbody.xml')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='C:/Users/Logan Koch/Github/haarcascade_frontalface_alt.xml')
parser.add_argument('--profile_cascade', help='Path to profile face cascade.', default='C:/Users/Logan Koch/Github/haarcascade_profileface.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='C:/Users/Logan Koch/Github/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()

body_cascade_name = args.body_cascade
face_cascade_name = args.face_cascade
profile_cascade_name = args.profile_cascade
eyes_cascade_name = args.eyes_cascade


body_cascade = cv.CascadeClassifier()
face_cascade = cv.CascadeClassifier()
profile_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()

#-- 1. Load the cascades
if not body_cascade.load(cv.samples.findFile(body_cascade_name)):
    print('--(!)Error loading body cascade')
    exit(0)
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not profile_cascade.load(cv.samples.findFile(profile_cascade_name)):
    print('--(!)Error loading profile cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)

#-- 2. Read the video stream
cap = cv.VideoCapture(0 + cv.CAP_DSHOW)

if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break

    detectAndDisplay(frame)
    if cv.waitKey(10) == 27:
        break
