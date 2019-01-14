import cv2,os
import numpy as np
from PIL import Image
import pickle

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(datasetPath):
    #get the path of all the files in the folder
    #create empth face list
    faceSamples_Train=[]
    # create empty ID list
    ids_Labels = []
    labelIds={}
    currentID = 0
    for root, dirs, files in os.walk(datasetPath):
        #now looping through all the image paths and loading the Ids and the images
        for file in files:
            imagePath = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            if not label in labelIds:
                labelIds[label] = currentID
                currentID+=1
            # getting the Id from the image
            Ids = labelIds[label]
            #loading the image and converting it to gray scale
            pilImage=Image.open(imagePath).convert('L')
            #Now we are converting the PIL image into numpy array
            imageNp=np.array(pilImage, 'uint8')
            # extract the face from the training image sample
            faces=detector.detectMultiScale(imageNp)
            #If a face is there then append that in the list as well as Id of it
            for (x,y,w,h) in faces:
                faceSamples_Train.append(imageNp[y:y + h, x:x + w])
                ids_Labels.append(Ids)
        with open("labels.pickle","wb") as f:
            pickle.dump(labelIds,f)
    return faceSamples_Train, ids_Labels


faceSamples_Train, ids_Labels = getImagesAndLabels('dataSets')
recognizer.train(faceSamples_Train, np.array(ids_Labels))
recognizer.save('trainner.yml')




