import cv2 as cv
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# def getImageAndLabels(path):
#     faceSamples=[]
#     ids=[]
#     face_detector = cv.CascadeClassifier('face_detection.xml')
#     categories = os.listdir(path)
#     for category in categories:
#         folder_path = os.path.join(path, category)
#         imageNames = os.listdir(folder_path)
#
#         for imageName in imageNames:
#             imagePath = os.path.join(folder_path, imageName)
#             image = cv.imread(imagePath)
#             faces = face_detector.detectMultiScale(image)
#
#             for x,y,w,h in faces:
#                 gray = cv.cvtColor(image[y:y+h, x:x+w], cv.COLOR_BGR2GRAY)
#                 resizedGray = cv.resize(gray, (60,60))
#                 faceSamples.append(resizedGray)
#                 if category == 'boys':
#                     ids.append(1)
#                 if category == 'girls':
#                     ids.append(0)
#     return faceSamples,ids
#
# path = 'D:/Machine Learning/mlproject/photoTrain'
# faces, ids = getImageAndLabels(path)
# x_test, y_test = getImageAndLabels('D:/Machine Learning/mlproject/photoTest')
# facesArray = np.array(faces) / 255.0
# idsArray = np.array(ids) # 104
# x_test = np.array(x_test) / 255.0
# y_test = np.array(y_test)
# #
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
#
# model.compile(
#     optimizer='adam',
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#     metrics=['sparse_categorical_accuracy']
# )
#
# model.fit(
#     facesArray,
#     idsArray,
#     batch_size=32,
#     epochs=500,
#     validation_data=(x_test,y_test),
#     validation_freq=1
# )
#
# model.summary()
#
from keras.models import  load_model

# model.save('gender_recognition.h5')

model = load_model('gender_recognition.h5')
face_classifier = cv.CascadeClassifier('face_detection.xml')
label_dict = {0:'Female', 1:'Male'}

#picturecheck
# img = cv.imread('D:/Machine Learning/mlproject/photoTest/boys/2.jpg')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# faces = face_classifier.detectMultiScale(gray)
# for (x,y,w,h) in faces:
#     face_img = gray[y:y+w, x:x+w]
#     resized = cv.resize(face_img, (60,60)) / 255.0
#     resized = np.reshape(resized, (1,60,60,1))
#     result = model.predict(resized)
#     label = np.argmax(result, axis=1)[0]
#     cv.rectangle(img, (x, y), ((x + w), (y + h)), (0, 255, 0), 5)
#     cv.putText(img, label_dict[label], (x, y-10), cv.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
# cv.imshow("gender recognition",img)
# cv.waitKey(0)

# realtime

window_name = "Real-time face recognition"
video = cv.VideoCapture(0)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    if len(faces) !=0:
        for (x,y,w,h) in faces:
            face_img = gray[y:y+w, x:x+w]
            resized = cv.resize(face_img, (60,60)) / 255.0
            resized = np.reshape(resized, (1,60,60,1))
            result = model.predict(resized)
            label = np.argmax(result, axis=1)[0]
            cv.rectangle(frame, (x, y), ((x + w), (y + h)), (0, 255, 0), 5)
            cv.putText(frame, label_dict[label], (x, y-10), cv.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    cv.imshow(window_name, frame)
    if cv.waitKey(1) == ord("q"):
        break

video.release()
cv.destroyAllWindows()
