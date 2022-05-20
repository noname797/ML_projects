# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 11:34:26 2019

@author: noname
"""

import tensorflow as tf
import numpy as np
from keras import backend as K
Mnist= tf.keras.datasets.mnist
(Train_imgs, train_labels),(test_imgs, test_labels)=Mnist.load_data ()
Train_imgs=Train_imgs.astype('float32')
test_imgs=test_imgs.astype('float32')
Train_imgs/=255
test_imgs/=255

img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    Train_imgs = Train_imgs.reshape(Train_imgs.shape[0], 1, img_rows, img_cols)
    test_imgs = test_imgs.reshape(test_imgs.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    Train_imgs = Train_imgs.reshape(Train_imgs.shape[0], img_rows, img_cols, 1)
    test_imgs = test_imgs.reshape(test_imgs.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

import keras
from keras.models import Sequential
from keras.layers import Flatten,Dense,Conv2D,MaxPooling2D

classifier=Sequential()

classifier.add(Conv2D(64,(3,3),activation='relu',input_shape=input_shape))
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D())
classifier.add(Flatten())
classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(128,activation='relu'))
'''upon adding more units/hidden layer we may see overfitting,as the train set accuracy goes upto one,test set/validation set accuracy
doesnt change much..
if we reduce the number of units (like 2-3only)..we see underfitting that is low training accuracy and poor test set accuracy as compared to other cases'''


classifier.add(Dense(10,activation='softmax'))

classifier.compile(optimizer='adam',metrics=['accuracy'],loss='sparse_categorical_crossentropy')

classifier.fit(Train_imgs,train_labels,epochs=10,validation_data=(test_imgs,test_labels))

y_pred=classifier.predict(test_imgs) #predicted multi output with probabilities

y_pred_num=np.argmax(y_pred,axis=1) #predicted single output

evaluation=classifier.evaluate(test_imgs,test_labels) #gives loss value and metrics

