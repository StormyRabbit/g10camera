import cv2 as cv
import os
import numpy as np
from PIL import Image

RECOGNIZER = cv.face.LBPHFaceRecognizer_create()
DETECTOR = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    images = [os.path.join(path,image) for image in os.listdir(path)]
    faceSamples = []
    Ids = []
    for image in images:
        #gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 
        bnw8bit = Image.open(image).convert('L') # convert 2 grayscale
        imageNp = np.array(bnw8bit, 'uint8') # convert 2 np.array
        imageId = int(os.path.split(image)[-1].split("_")[1].split(".")[0])
        faces = DETECTOR.detectMultiScale(imageNp)
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(imageId)

    return faceSamples, Ids


faces, Ids = getImagesAndLabels('./users/marcus/')
RECOGNIZER.train(faces, np.array(Ids))
RECOGNIZER.save("marcus.xml")

