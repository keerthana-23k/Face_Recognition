import cv2
import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
face_id=input('enter your id')
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

face_detector = cv2.CascadeClassifier('C:\\Users\\keert\\Downloads\\Face Recoginition\\Face Recoginition\\haarcascade_frontalface_default.xml')

count = 0

assure_path_exists("dataset/")

while(True):

    _, image_frame = video.read()

    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (0,0,0), 2)
        cv2.imshow('dataset_capture', image_frame)
        count += 1
        
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

    if count>=10:
        print("Successfully Captured")
        break

video.release()

