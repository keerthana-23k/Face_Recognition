import cv2, numpy as np;
import xlwrite,firebase_admin as fire;
import time
import sys
start=time.time()
period=8
face_cas = cv2.CascadeClassifier('C:\\Users\\keert\\Desktop\\Face_recog\\attendance\\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0);
recognizer = cv2.face.LBPHFaceRecognizer_create();
recognizer.read('C:\\Users\\keert\\Desktop\\trainer\\trainer.yml');
flag = 0;
id=0;
filename='filename';
dict = {
            'item1': 1
}
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, img = cap.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    faces = face_cas.detectMultiScale(gray, 1.3, 7);
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),2);
        id,conf=recognizer.predict(roi_gray)
        if(conf > 50):
         if(id==1):
            id='Keerthana'
            if((str(id)) not in dict):
                filename=xlwrite.output('attendance','class1',1,id,'yes');
                dict[str(id)]=str(id);
         elif(id==2):
            id='Komathi'
            if ((str(id)) not in dict):
                filename =xlwrite.output('attendance', 'class1', 2, id, 'yes');
                dict[str(id)] = str(id);
         elif(id==3):
            id='Kaushii'
            if ((str(id)) not in dict):
                filename =xlwrite.output('attendance', 'class1', 2, id, 'yes');
                dict[str(id)] = str(id);
         elif(id==5):
            id='Prashanth'
            if ((str(id)) not in dict):
                filename =xlwrite.output('attendance', 'class1', 2, id, 'yes');
                dict[str(id)] = str(id);
         elif(id==6):
            id='Sumathi'
            if ((str(id)) not in dict):
                filename =xlwrite.output('attendance', 'class1', 2, id, 'yes');
                dict[str(id)] = str(id);
         elif(id==7):
            id='Krishna'
            if ((str(id)) not in dict):
                filename =xlwrite.output('attendance', 'class1', 2, id, 'yes');
                dict[str(id)] = str(id);
         elif(id==8):
            id='Kannan'
            if ((str(id)) not in dict):
                filename =xlwrite.output('attendance', 'class1', 3, id, 'yes');
                dict[str(id)] = str(id);
         elif(id==20):
            id='Abinaya'
            if ((str(id)) not in dict):
                filename =xlwrite.output('attendance', 'class1', 2, id, 'yes');
                dict[str(id)] = str(id);
        else:
             id = 'Unknown'
             flag=flag+1
             break
        
        cv2.putText(img,str(id),(x,y-10),font,0.55,(120,255,120),1)
        #cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,(0,0,255));
    cv2.imshow('frame',img);
    #cv2.imshow('gray',gray);
    if flag == 10:
        print("Transaction Blocked")
        break;
    if time.time()>start+period:
        break;
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
    print('attendance successfully done')
cap.release();
cv2.destroyAllWindows();
