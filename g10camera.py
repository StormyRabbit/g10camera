"""
Lasse Sjöblom och Tove Silfver de Verdier
"""
import argparse
import shutil
import cv2 as cv
import numpy as np
from PIL import Image
import os
import time
ARG_PARSER = argparse.ArgumentParser()
ARG_PARSER.add_argument("--adduser", help="Add a new user by gathering photos")

ARGS = ARG_PARSER.parse_args()

CAM = cv.VideoCapture(0)

DETECTOR = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
RECOGNIZER = cv.face.LBPHFaceRecognizer_create()
USER_PATH = "images/"
USER_XML_PATH = "users"
PICTURES_TO_TAKE = 100

if not os.path.exists(USER_XML_PATH):
    os.makedirs(USER_XML_PATH)

if not os.path.exists(USER_PATH):
    os.makedirs(USER_PATH)


class AddUser:
    def __init__(self):
        print("hello {}".format(ARGS.adduser))
        self.picture = 0
        self._gather_data()
        self._train_system()

    def _train_system(self):
        faceData = []
        ids = []
        for folder in os.listdir(USER_PATH):
            if os.path.isdir(USER_PATH + folder):
                for image in os.listdir(USER_PATH + folder):
                    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                    imageNp = np.array(gray, 'uint8')
                    imageId = int(os.path.split(image[-1].split("_")[0].split(".")[0]))
                    faces = DETECTOR.detectMultiScale(imageNp)
                    for (x,y,w,h) in faces:
                        faceData.append(imageNp[y:y+h,x:x+w])
                        ids.append(imageId)
                RECOGNIZER.train(faces, np.array(Ids))
                RECOGNIZER.save(image.split("_")[0], ".xml")
                shutil.remtree(TRAINING_DATA_PATH + folder, ignore_errors=True)

    def _gather_data(self):
        while(self.picture < PICTURES_TO_TAKE):
            ret, img = CAM.read()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = DETECTOR.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
                img_path = USER_PATH + ARGS.adduser + "/" + ARGS.adduser + "_" + str(self.picture) + ".jpeg"
                cv.imwrite(img_path, gray[y:y+h, x:x+w])
                self.picture += 1
            cv.imshow('data_gathering', img)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        CAM.release()
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
                cv.putText(img, "text", (x,y), self._font, self._font_scale, self._font_color, self._line_type)
            cv.imshow("camera", img)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break


def loadUsers():
    for xml_file in os.listdir(USER_XML_PATH):
        print("Loading {}".format(xml_file))
        RECOGNIZER.read(USER_XML_PATH + xml_file)


if __name__ == '__main__':
    if ARGS.adduser:
        AddUser()
    loadUsers()
    camera().start()
