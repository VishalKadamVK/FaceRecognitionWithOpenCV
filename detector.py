import cv2
import numpy as np
import time
import pickle
# import speech

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
originalLabels = {}
with open("labels.pickle", "rb") as f:
    originalLabels = pickle.load(f)
    labels = {v:k for k, v in originalLabels.items()}


cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX
presentPeaople = []
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        IDs = "Unknown"
        if(conf<50):
            IDs = labels[Id]
        cv2.putText(im, IDs, (x, y+h), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # if IDs != "Unknown":
        #     presentPeaople.append(IDs)
        #     speech.say('Hi {}, Welcome in Meeting'.format(IDs))
    cv2.imshow('im', im)
    if (cv2.waitKey(1) == ord('q')):
        break
cam.release()
cv2.destroyAllWindows()
