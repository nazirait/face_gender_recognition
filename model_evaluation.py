#!/usr/bin/env python
# coding: utf-8

# In[19]:


import cv2 as cv
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from keras.models import  load_model


# In[20]:


#import the trained model
model = load_model('gender_recognition.h5')
face_classifier = cv.CascadeClassifier('face_detection.xml')


# In[21]:


#evaluate the model
girls_TP=0
girls_FN=0
girls_FP=0
girls_TN=0
boys_TP=0
boys_FN=0
boys_FP=0
boys_TN=0
Nb_boys=0
Nb_girls=0
path = 'C:/Users/hugol/Downloads/Telegram Desktop/mlproject/mlproject/photo/phototest'
categories = os.listdir(path)
for category in categories:
    folder_path = os.path.join(path, category)
    imageNames = os.listdir(folder_path)
    for imageName in imageNames:
        imagePath = os.path.join(folder_path, imageName)
        img = cv.imread(imagePath)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)
        for (x,y,w,h) in faces:
            face_img = gray[y:y+w, x:x+w]
            resized = cv.resize(face_img, (60,60)) / 255.0
            resized = np.reshape(resized, (1,60,60,1))
        result = model.predict(resized)
        label = np.argmax(result, axis=1)[0]
        if(category=='girls'):
            if(label==0):
                girls_TP+=1
                boys_TN+=1
            else:
                girls_FN+=1
                boys_FP+=1
            Nb_girls+=1
        elif(category=='boys'):
            if(label==1):
                boys_TP+=1
                girls_TN+=1
            else:
                boys_FN+=1
                girls_FP+=1
            Nb_boys+=1




# In[22]:


#print the result
print("boys TP: " + str(boys_TP))
print("boys FN: " + str(boys_FN))
print("boys FP: " + str(boys_FP))
print("boys TN: " + str(boys_TN))

#accuracy=number of correct prediction/total Number of prediction
girls_accuracy=girls_TP/Nb_girls
boys_accuracy=boys_TP/Nb_boys
print("girls accuracy: "+ str(girls_accuracy))
print("boys accuracy: "+ str(boys_accuracy ))

accuracy= (girls_TP + boys_TP)/(Nb_girls+Nb_boys)
print("Overall accuracy: "+ str(accuracy))

#precision=TP/TP+FP
girls_precision = girls_TP/(girls_TP+girls_FP)
boys_precision = boys_TP/(boys_TP+boys_FP)
print("girls precision: "+ str(girls_precision) )
print("boys precision: "+ str(boys_precision) )

precision= (boys_TP + girls_TP)/(boys_TP + girls_TP +girls_FP+boys_FP)
print("Overall precision: "+ str(precision))

#recall=TP/TP+FN
girls_recall = girls_TP/(girls_TP+girls_FN)
boys_recall = boys_TP/(boys_TP+boys_FN)
print("girls recall: "+ str(girls_recall) )
print("boys recall: "+ str(boys_recall) )

recall = (boys_TP + girls_TP)/(boys_TP + girls_TP +girls_FN+boys_FN)
print("Overall recall: "+ str(recall))

#F1_score=2*(precision*recall)/(precision+recall)
girls_F1_score = 2*(girls_precision*girls_recall)/(girls_precision+girls_recall)
boys_F1_score = 2*(boys_precision*boys_recall)/(boys_precision+boys_recall)
print("girls F1_score: "+ str(girls_F1_score) )
print("boys F1_score: "+ str(boys_F1_score) )

F1_score = 2*(precision*recall)/(precision+recall)
print("F1 score: "+ str(F1_score))


# In[23]:


binary = np.array([[boys_TP, boys_FP],
                   [boys_FN, boys_TN]])

fig, ax = plot_confusion_matrix(conf_mat=binary,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True,
                               class_names=['boys','girls'])
plt.show()

