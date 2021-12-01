"""
Created on Mon Jul 6 10:15:22 2020

@author: Aurelien Callens

This script regroups several functions useful for the CNN training :
    - 2 functions that generates CNN architectures that are not implemented
    in keras function.
    - 1 function that compute the confusion matrix and other metrics
"""
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from pycm import *


def create_base_model(output=3):
    """
    Generate the "custom" CNN architecture
    :param num ouptut: number of output classes
    """
    model = Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2,
                            activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(output, activation='softmax'))
    return(model)


def create_alex_model(output=3):
    """
    Generate the AlexNet CNN architecture

    :param num ouptut: number of output classes
    """
    alex_model = Sequential()

    # 1st Convolutional Layer
    alex_model.add(layers.Conv2D(filters=96, input_shape=(224, 224, 3),
                                 kernel_size=(11, 11), strides=(4, 4),
                                 padding="valid", activation='relu'))
    # Max Pooling
    alex_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                                       padding="valid"))

    # 2nd Convolutional Layer
    alex_model.add(layers.Conv2D(filters=256, kernel_size=(11, 11),
                                 strides=(1, 1), padding="valid",
                                 activation='relu'))
    # Max Pooling
    alex_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                                       padding="valid"))

    # 3rd Convolutional Layer
    alex_model.add(layers.Conv2D(filters=384, kernel_size=(3, 3),
                                 strides=(1, 1), padding="valid",
                                 activation='relu'))

    # 4th Convolutional Layer
    alex_model.add(layers.Conv2D(filters=384, kernel_size=(3, 3),
                                 strides=(1, 1), padding="valid",
                                 activation='relu'))

    # 5th Convolutional Layer
    alex_model.add(layers.Conv2D(filters=256, kernel_size=(3, 3),
                                 strides=(1, 1), padding="valid",
                                 activation='relu'))
    # Max Pooling
    alex_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                                       padding="valid"))

    # Passing it to a Fully Connected layer
    alex_model.add(layers.Flatten())
    # 1st Fully Connected Layer
    alex_model.add(layers.Dense(4096, activation='relu'))
    # Add Dropout to prevent overfitting
    alex_model.add(layers.Dropout(0.5))
    # 2nd Fully Connected Layer
    alex_model.add(layers.Dense(4096, activation='relu'))
    # Add Dropout
    alex_model.add(layers.Dropout(0.5))

    # Output Layer
    alex_model.add(layers.Dense(output, activation='softmax'))

    return(alex_model)


def make_conf_mat(model, test_gen, train_gen, threshold=False, print=True):
    """
    Compute the confusion matrix and several metrics with pycm

    :param model: Keras model trained
    :param test_gen: Test generator on which to compute the confusion matrix
    :param train_gen: Train generator only used when 'threshold = True' to
    compute the prior probabilities for each class
    :param threshold: If True perform the thresholding method.
    :param print: If True print the confusion matrix
    """
    predict = model.predict(test_gen, steps=test_gen.n)

    if(threshold):
        prob = pd.value_counts(train_gen.classes)/train_gen.n
        predict = pd.DataFrame(predict)
        predict = predict/prob
        predict = (predict.T / predict.T.sum()).T
        predict = np.array(predict)

    y_classes = predict.argmax(axis=-1)
    cm = ConfusionMatrix(actual_vector=test_gen.classes.tolist(),
                         predict_vector=y_classes.tolist())

    if(print):
        cm.print_matrix()
        cm.stat(summary=True)
    else:
        return(cm)
