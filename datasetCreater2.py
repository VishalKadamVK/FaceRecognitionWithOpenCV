import cv2
import os
import numpy as np
dataSets = 'dataSets'
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

Label = raw_input('Enter user Label: ')
os.mkdir("{}//{}//{}".format(os.getcwd(), dataSets, Label))
sampleNum = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        sampleNum = sampleNum + 1
        cv2.imwrite("{}/{}/User.".format(dataSets, Label)+str(sampleNum)+".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.waitKey(100)
    cv2.imshow("Face", img)
    cv2.waitKey(1)
    if sampleNum>1000:
        break
cam.release()
cv2.destroyAllWindows()

