import cv2 as cv
import argparse
import os
CAM = cv.VideoCapture(0)
DETECTOR = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
PICTURES = 0
PARSER = argparse.ArgumentParser()
PARSER.add_argument("user_name", help="User name for new user")
ARGS = PARSER.parse_args()
user_name = ARGS.user_name

USER_PATH = "./users/" + user_name + "/"

if not os.path.exists(USER_PATH):
    os.makedirs(USER_PATH)


while(True):
    ret, img = CAM.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = DETECTOR.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0,),2)
        cv.imwrite(USER_PATH + user_name + "_" + str(PICTURES) + ".jpeg", gray[y:y+h,x:x+w])
        PICTURES += 1
    cv.imshow('frame', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    elif PICTURES > 10:
        print("DATA GATHERED")
        break

CAM.release()
cv.destroyAllWindows()
