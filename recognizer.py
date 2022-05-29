import cv2
import numpy as np
import face_recognition


imgelon = face_recognition.load_image_file('elon.jpg')
imgelon_rgb = cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB)
#----------Finding face Location for drawing bounding boxes-------
face = face_recognition.face_locations(imgelon_rgb)[0]
copy = imgelon_rgb.copy()
#-------------------Drawing the Rectangle-------------------------
cv2.rectangle(copy, (face[3], face[0]),(face[1], face[2]), (255,0,255), 2)
cv2.waitKey(0)
train_encode = face_recognition.face_encodings(imgelon_rgb)[0]

test = face_recognition.load_image_file('elon_5.jpg')
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
test_encode = face_recognition.face_encodings(test)[0]
print(face_recognition.compare_faces([train_encode],test_encode))