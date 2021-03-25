#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 6 10:15:22 2020

@author: Aurelien Callens 

This script regroups several functions useful for the CNN training : 
    - 2 functions that generates CNN architectures that are not implemented in keras function.
    - 1 function that compute the confusion matrix and other metrics 
"""

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D


def create_base_model(output = 3):
    """
    Generate the "custom" CNN architecture 
    
    :param num ouptut: number of output classes
    
    """
    model = Sequential()
    
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), strides = 2, activation='relu', input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output, activation='softmax')) 
    return(model)

def create_alex_model(output = 3):
    """
    Generate the AlexNet CNN architecture 
    
    :param num ouptut: number of output classes
    
    """
    
    alex_model = Sequential()
    
    # 1st Convolutional Layer
    alex_model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding="valid", activation = 'relu'))
    # Max Pooling
    alex_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))
    
    # 2nd Convolutional Layer
    alex_model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding="valid", activation = 'relu'))
    # Max Pooling
    alex_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))
    
    # 3rd Convolutional Layer
    alex_model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid", activation = 'relu'))
    
    # 4th Convolutional Layer
    alex_model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid", activation = 'relu'))
    
    # 5th Convolutional Layer
    alex_model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="valid", activation = 'relu'))
    # Max Pooling
    alex_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))
    
    # Passing it to a Fully Connected layer
    alex_model.add(Flatten())
    # 1st Fully Connected Layer
    alex_model.add(Dense(4096, activation = 'relu'))
    # Add Dropout to prevent overfitting
    alex_model.add(Dropout(0.5))
    
    # 2nd Fully Connected Layer
    alex_model.add(Dense(4096, activation = 'relu'))
    alex_model.add(Activation("relu"))
    # Add Dropout
    alex_model.add(Dropout(0.5))

    
    # Output Layer
    alex_model.add(Dense(output, activation='softmax'))
    return(alex_model)




from pycm import *
def make_conf_mat(model, test_gen, print = True):
    """
    Compute the confusion matrix and several metrics with pycm 
    
    :param model: Keras model trained
    :param test_gen: Test generator on which to compute the confusion matrix 
    :param print: If True print the confusion matrix
    
    """
    predict = model.predict(test_gen, steps = test_gen.n)
    
    y_classes = predict.argmax(axis=-1)
       
    cm = ConfusionMatrix(actual_vector=test_gen.classes.tolist(), predict_vector=y_classes.tolist())
    
    if(print):
        cm.print_matrix()
        cm.stat(summary = True)
    else:
        return(cm)
    
    
    