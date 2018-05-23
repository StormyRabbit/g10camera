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
# constants for handeling adduser argument
ARG_PARSER = argparse.ArgumentParser()
ARG_PARSER.add_argument("--adduser", help="Add a new user by gathering photos")
ARGS = ARG_PARSER.parse_args()
# constant for webcam
CAM = cv.VideoCapture(0)
# constant for different file and directory locations
USER_PATH = "images/"
TRAINING_DATA = "data.xml"
USERS = {}
# load different file data needed for 
if os.path.isfile('user_details.json'):
    with open('user_details.json') as user_data:
        USERS = json.load(user_data)
DETECTOR = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
RECOGNIZER = cv.face.LBPHFaceRecognizer_create()
if os.path.isfile(TRAINING_DATA):
    RECOGNIZER.read(TRAINING_DATA)
PICTURES_TO_TAKE = 20

# if any directory is missing, create it
if not os.path.exists(USER_PATH):
    os.makedirs(USER_PATH)

class AddUser:
    """
    Class for adding new users, responsible for taking pictures and training the recognizer.
    Should not be used as an object, takes parameters from ARG_PARSER.
    """

    def __init__(self):
        """Constructor for AddUser class, calls the functions needed to add a new user"""
        self.picture = 0
        self._gather_data()
        self._train_system()

    def _train_system(self):
        """Trains the system with all the image data in the USER_PATH file structure"""
        user_id = len(USERS)
        faceData = []
        ids = []
        for folder in os.listdir(USER_PATH):
            if os.path.isdir(USER_PATH + folder):
                for image in os.listdir(USER_PATH + folder): 
                    gray = Image.open(USER_PATH + folder + "/" + image).convert('L') # convert image to grayscale
                    imageNp = np.array(gray, 'uint8') # flatten grayscale image to numpy array.
                    faces = DETECTOR.detectMultiScale(imageNp, 1.1, 10) # find all faces, will be none if none is found
                    for (x,y,w,h) in faces: 
                        faceData.append(imageNp[y:y+h,x:x+w])
                        ids.append(user_id)

                # update the recognizer model with new data
                # and update the user manifesto with the new user id and name.
                # finally remove all user related images.
                RECOGNIZER.update(faceData, np.array(ids))
                USERS[str(user_id)] = folder  
                with open('user_details.json', 'w') as out:
                    json.dump(USERS, out)
                RECOGNIZER.save(TRAINING_DATA)
                shutil.rmtree(USER_PATH + folder, ignore_errors=True)

    def _gather_data(self):
        """
            Function to gather and store all the data used for training.
            In order to provide good training data only one face should be present during
            data gathering.
        """
        if not os.path.exists(ARGS.adduser):
            os.makedirs(USER_PATH + ARGS.adduser)

        while self.picture < PICTURES_TO_TAKE:
            ret, img = CAM.read()
            # convert to grayscale to allow for facial detection
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
            faces = DETECTOR.detectMultiScale(gray, 1.1, 10)
            for (x, y, w, h) in faces:
                cv.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
                img_path = USER_PATH + ARGS.adduser + "/" + ARGS.adduser + "_" + str(self.picture) + ".png"
                # store the image grayscale, limited to only the face part
                cv.imwrite(img_path, gray[y:y+h, x:x+w])
                self.picture += 1
            cv.imshow('Data Gathering', img)
            # wait for 100 ms to allow for the user to change angle
            if cv.waitKey(100) & 0xFF == ord('q'):
                break
        cv.destroyAllWindows()

class camera:
    """
        Main class for the system, responsible for running the actual loop
        that captures the video stream, also calls the detection functions
        and displays the on screen elements when a person is detected.
    """
    def __init__(self):
        """
        Constructor, only used for initializing some variables to provide cleaner code.
        """
        self._font = cv.FONT_HERSHEY_SIMPLEX
        self._font_scale = 1
        self._font_color = (255, 255, 255) # NOTE: color system is BGR not RGB!
        self._line_type = 2
        self._rect_color = None

    def start(self):
        """
            Main function of the camera system, continues to 
            capture the video stream until q is entered. 
            Will put up the user name of people with over 50% confidence.
            If confidence is lower the 50% the user will be classified as unauthorized.
        """
        while True:
            ret, img = CAM.read()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = DETECTOR.detectMultiScale(gray, 1.1, 10)
            text = ""
            for(x, y, w, h) in faces:
                user, conf = RECOGNIZER.predict(gray[y:y+h, x:x+w])
                # convert the wierd conf system used by openCV to actual %, higher == better.
                conf = 100 - round(float(conf),2)
                print("USER: {} \n CONF: {}".format(user, conf))
                if conf > 50:
                    try:
                        text = "{} CONF: {}%".format(USERS[str(user)], conf)
                        self._rect_color = (0, 255, 0) # set rect color to green if the user is authorized
                    except:
                        self._rect_color = (0, 0, 255) # set color to RED
                        text = "unauthorized" 
                else:
                    self._rect_color = (0, 0, 255)
                    text = "unknown" 
                # display the text and rectangle over the faces of the detected individuals.
                cv.rectangle(img, (x,y), (x+w, y+h), self._rect_color, 2)
                cv.putText(img, text, (x,y), self._font, self._font_scale, self._font_color, self._line_type)
            cv.imshow("camera", img)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
        # when while loop is over, release cam
        CAM.release()

if __name__ == '__main__':
    if ARGS.adduser:
        AddUser()
    camera().start()
    
