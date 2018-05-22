import cv2 as cv
import numpy as np

RECOGNIZER = cv.face.LBPHFaceRecognizer_create()
#RECOGNIZER.read('user_xml/marcus.xml')
RECOGNIZER.read('user_xml/lasse.xml')

HAARCASCADE = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

CAM = cv.VideoCapture(0)

FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_COLOR = (255, 255, 255)
LINE_TYPE = 2

while True:
    ret, img = CAM.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = HAARCASCADE.detectMultiScale(gray, 1.2, 5)
    
    for(x, y, w, h) in faces:
        cv.rectangle(img, (x-50, y-50), (x+w+50, y+h+50), (225,0,0),2)
        user, conf = RECOGNIZER.predict(gray[y:y+h, x:x+w])
        print("USER: #{} \n CONF: #{}".format(user, conf))
        cv.putText(img, "HEJ",(x,y), FONT, FONT_SCALE, FONT_COLOR, LINE_TYPE)


    cv.imshow('img', img)
    if cv.waitKey(10) & 0xFF == ord('Q'):
        break
