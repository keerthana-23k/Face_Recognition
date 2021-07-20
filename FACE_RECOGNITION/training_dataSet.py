import os
import cv2
import numpy as np
from PIL import Image
r = cv2.face.LBPHFaceRecognizer_create()
d = cv2.CascadeClassifier('C:\\Users\\keert\\Desktop\\Face_recog\\attendance\\haarcascade_frontalface_default.xml');

def paths(path):

    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]

    Samples=[]

    Ids=[]

    for imagePath in imagePaths:

        pilImage=Image.open(imagePath).convert('L')

        imageNp=np.array(pilImage,'uint8')

        Id=int(os.path.split(imagePath)[-1].split(".")[1])

        faces=d.detectMultiScale(imageNp)

        for (x,y,w,h) in faces:
            Samples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return Samples,Ids

faces,Ids = paths('C:\\Users\\keert\\Desktop\\Face_recog\\attendance\\dataset')
s = r.train(faces, np.array(Ids))
print("Successully trained")
r.save('C:\\Users\\keert\\Desktop\\trainer\\trainer.yml')



