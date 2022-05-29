import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle


images = []
classNames = []
mylist = os.listdir("C:/Users/aadwa/OneDrive/Documents/GitHub/Face_Recog/face_storage")
for cl in mylist:
    curImg = cv2.imread(f'{"C:/Users/aadwa/OneDrive/Documents/GitHub/Face_Recog/face_storage"}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList
encoded_face_train = findEncodings(images)

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'n{name}, {time}, {date}')

markAttendance("elon")
markAttendance("jeff")
