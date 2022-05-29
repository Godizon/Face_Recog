import cv2
import face_recognition
import os

images = []
classNames = []
mylist = os.listdir("C:/Users/aadwa/OneDrive/Documents/GitHub/Face_Recog/face_storage")
for cl in mylist:
    curImg = cv2.imread(f'C:/Users/aadwa/OneDrive/Documents/GitHub/Face_Recog/face_storage/{cl}')
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

test = face_recognition.load_image_file('test.jpg')
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
test_encode = face_recognition.face_encodings(test)[0]
i=0
for enc in encoded_face_train:
    if(face_recognition.compare_faces([enc] ,test_encode)[0]) :
        print(classNames[i])
    i+=1