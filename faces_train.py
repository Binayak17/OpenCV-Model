import os
import numpy as np 
import cv2 as cv 

people =['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Mindy Kaling', 'Madonna']
DIR = '/Users/binayak/ML4E/Face Recognition model/Faces/train'

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# cv2_base_dir = os.path.dirname(os.path.abspath(cv.__file__))
# haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
# haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done ------------------')

# print(f'Length of features : {len(features)}')
# print(f'Length of labels : {len(labels)}')

features = np.array(features, dtype = 'object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

############## Train ####################

face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')

# np.save('features.npy', features)
# np.save('labels.npy', labels)
