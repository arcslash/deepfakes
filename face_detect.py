import cv2
import numpy as np

#cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#loading images
img = cv2.imread('nicholas_cage.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for mat in faces:
    print(mat)

    cv2.rectangle(img, (mat[0], mat[1]), (mat[0] + mat[2], mat[1] + mat[3]), (255, 0, 0), 5)

#print(faces)

cv2.imshow('image',img)

cv2.waitKey(0)
