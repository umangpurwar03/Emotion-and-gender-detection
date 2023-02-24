import cv2
import numpy as np
from deepface import DeepFace
import streamlit

face_cascade=cv2.CascadeClassifier(r'C:\Users\umang\Downloads\face emotion detection\opencv\data\haarcascades\haarcascade_frontalface_alt2.xml')
cap=cv2.VideoCapture(0)
scaling_factor=1
while True :
    ret, frame =cap.read()
    frame=cv2.resize(frame,None,fx=scaling_factor,fy=scaling_factor,interpolation=cv2.INTER_AREA)
    faces=face_cascade.detectMultiScale(frame,scaleFactor=1.3, minNeighbors=3)
    for(x,y,w,h) in faces:
        face=frame[y:y+h,x:x+w]
        emotions=DeepFace.analyze(face, actions=['emotion'],enforce_detection=False)
        gender=DeepFace.analyze(face, actions=['gender'],enforce_detection=False)

        emotion_text='emotion: '+ emotions[0]['dominant_emotion']
        gender_text='gender: '+ gender[0]['dominant_gender']
        print(gender_text)
        print(emotion_text)
        cv2.putText(frame, emotion_text,(x, y-20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        cv2.putText(frame, gender_text,(x, y-40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1)
    if key != -1 and key == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
