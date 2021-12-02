#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from timeit import default_timer as timer
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
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


def create_incep_model(output=3, weights_cnn=None):
    """
    Generate the Inception CNN architecture
    :param num ouptut: number of output classes
    :param weights_cnn: None or imagenet'
    """
    # Create the model
    incep_model = InceptionV3(include_top=False, pooling='max',
                              weights=weights_cnn, input_shape=(224, 224, 3))
    model_incep = Sequential()
    model_incep.add(incep_model)
    # Add new layers
    model_incep.add(layers.Flatten())
    model_incep.add(layers.Dense(512, activation='relu'))
    model_incep.add(layers.Dropout(0.5))
    model_incep.add(layers.Dense(output, activation='softmax'))

    return(incep_model)


def create_vgg16_model(output=3, weights_cnn=None):
    """
    Generate the VGG16 CNN architecture
    :param num ouptut: number of output classes
    :param weights_cnn: None or imagenet'
    """
    vgg_model = VGG16(include_top=False, pooling='max', weights=weights_cnn,
                      input_shape=(224, 224, 3))

    # Create the model
    model_vgg = Sequential()
    model_vgg.add(vgg_model)
    # Add new layers
    model_vgg.add(layers.Flatten())
    model_vgg.add(layers.Dense(512, activation='relu'))
    model_vgg.add(layers.Dropout(0.5))
    model_vgg.add(layers.Dense(output, activation='softmax'))

    return(model_vgg)


def generator_init(img_path, batch_size=32):

    # Declare data augmentation techniques :
    train_datagen = image.ImageDataGenerator(rescale=(1/255),
                                             vertical_flip=True,
                                             channel_shift_range=50.0,
                                             brightness_range=(0.5, 1.5),
                                             height_shift_range=0.2)

    test_datagen = image.ImageDataGenerator(rescale=(1/255))

    # Create the generators

    train_generator = train_datagen.flow_from_directory(
        directory=img_path + "/train/",
        color_mode="rgb",
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42)

    train_ov_generator = train_datagen.flow_from_directory(
        directory=img_path + "/train_ov/",
        color_mode="rgb",
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42)

    val_generator = test_datagen.flow_from_directory(
        directory=img_path + "/val/",
        color_mode="rgb",
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42)

    test_generator = test_datagen.flow_from_directory(
        directory=img_path + "/test/",
        color_mode="rgb",
        target_size=(224, 224),
        batch_size=1,
        class_mode="categorical",
        shuffle=False,
        seed=42)
    return(train_generator, train_ov_generator, val_generator, test_generator)


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

class StepDecay():
    def __init__(self, initAlpha=0.01, factor=0.1, dropEvery=10):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery

    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = self.initAlpha * (self.factor ** exp)
        # return the learning rate
        return float(alpha)


# Callback to record training time
class TimingCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

