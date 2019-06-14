import cv2

import numpy as np


rec= cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('/home/pratik/PycharmProjects/ml/venv/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
rec.read('/home/pratik/trainner/trainner.yml')
Ids=0
font = cv2.FONT_HERSHEY_SIMPLEX

while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        Ids, conf = rec.predict(gray[y:y+h,x:x+w])
        if (conf>30):
            if (Ids==3):
               Ids= "Pratik"



            if (Ids==5):
                Ids= "didi"

        cv2.putText(img,str(Ids), (x,y+h),font,2,(0,255,0), 2)
    cv2.imshow('frame', img)
    cv2.waitKey(100)
cap.release()
cv.destroyAllWindows()
