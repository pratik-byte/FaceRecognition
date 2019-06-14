import numpy as np
import cv2 as cv
face_cascade = cv.CascadeClassifier('/home/pratik/PycharmProjects/ml/venv/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')
cam = cv.VideoCapture(0)

id = input('enter user id')
sampleNum = 0
while(True):
    ret, img = cam.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY);
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        sampleNum = sampleNum + 1
        cv.imwrite('/home/pratik/dataset/'+str(id)+'.'+str(sampleNum)+'.jpg', gray[y:y+h,x:x+w])
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv.imshow('frame', img)
        cv.waitKey(1)
        if sampleNum>20:
            break
cap.release()
cv.destroyAllWindows()
