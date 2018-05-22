"""
Lasse Sjöblom och Tove Silfver de Verdier
"""
import argparse
import shutil
import cv2 as cv
import numpy as np
from PIL import Image
import os
import json
ARG_PARSER = argparse.ArgumentParser()
ARG_PARSER.add_argument("--adduser", help="Add a new user by gathering photos")

USER_PATH = "images/"
XML_FILE = "user_info.xml"
ARGS = ARG_PARSER.parse_args()

CAM = cv.VideoCapture(0)
USERS = {}
if os.path.isfile('user_details.json'):
    with open('user_details.json') as user_data:
        USERS = json.load(user_data)

DETECTOR = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
RECOGNIZER = cv.face.LBPHFaceRecognizer_create()
PICTURES_TO_TAKE = 100

if not os.path.exists(USER_PATH):
    os.makedirs(USER_PATH)

if os.path.isfile(XML_FILE):
    RECOGNIZER.read(XML_FILE)
    
class AddUser:
    def __init__(self):
        self.picture = 0
        self._gather_data()
        self._train_system()

    def _train_system(self):
        user_id = len(USERS)
        faceData = []
        ids = []
        for folder in os.listdir(USER_PATH):
            if os.path.isdir(USER_PATH + folder):
                for image in os.listdir(USER_PATH + folder):
                    gray = Image.open(USER_PATH + folder + "/" + image).convert('L') 
                    imageNp = np.array(gray, 'uint8')
                    imageId = image.split("_")[0]
                    faces = DETECTOR.detectMultiScale(imageNp)
                    for (x,y,w,h) in faces:
                        faceData.append(imageNp[y:y+h,x:x+w])
                        ids.append(user_id)

                RECOGNIZER.train(faceData, np.array(ids))
                USERS[user_id] = folder 
                with open('user_details.json', 'w') as out:
                    json.dump(USERS, out)
                RECOGNIZER.save(XML_FILE)
                shutil.rmtree(USER_PATH + folder, ignore_errors=True)

    def _gather_data(self):
        if not os.path.exists(USER_PATH + ARGS.adduser +"/"):
            os.makedirs(USER_PATH + ARGS.adduser +"/")
        while self.picture < PICTURES_TO_TAKE:
            ret, img = CAM.read()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = DETECTOR.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
                img_path = USER_PATH + ARGS.adduser + "/" + ARGS.adduser + "_" + str(self.picture) + ".png"
                cv.imwrite(img_path, gray[y:y+h, x:x+w])
                self.picture += 1
            cv.imshow('data_gathering', img)
            if cv.waitKey(100) & 0xFF == ord('q'):
                break
        cv.destroyAllWindows()

class camera:
    def __init__(self):
        self._font = cv.FONT_HERSHEY_SIMPLEX
        self._font_scale = 1
        self._font_color = (255, 255, 255)
        self._line_type = 2

    def start(self):
        while True:
            ret, img = CAM.read()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = DETECTOR.detectMultiScale(gray, 1.2, 5)
            for(x, y, w, h) in faces:
                user, conf = RECOGNIZER.predict(gray[y:y+h, x:x+w])
                print("USER: {} \n CONF: {}".format(user, conf))
                cv.putText(img, USERS[str(user)], (x,y), self._font, self._font_scale, self._font_color, self._line_type)
            cv.imshow("camera", img)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
        CAM.release()

if __name__ == '__main__':
    if ARGS.adduser:
        AddUser()
    camera().start()
    
