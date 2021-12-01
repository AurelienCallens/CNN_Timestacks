#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 25 2021

@author: Aurelien Callens

This script trains different CNN models and saves them
"""

import pickle
import numpy as np

from timeit import default_timer as timer
from sklearn.utils import class_weight
from numpy.random import seed
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.preprocessing import image
from scripts.Cnn_functions import *
from sklearn.utils import class_weight
from scripts.Make_graphs import plot_image, plot_value_array, make_train_graphs
from scripts.Cnn_functions import make_conf_mat


# Setting seeds for reproducibility
seed(1)
# tensorflow.random.set_random_seed(2)

# Gpu loading

"""
from tensorflow.keras.backend import set_session
config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.80
sess = tensorflow.Session(config=config)
set_session(sess)
"""

# Functions
# # Initialize generators

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


# # Callbacks for CNN training

class StepDecay(LearningRateDecay):
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


# Analysis
# # Generator initialization
batch_size = 32
train_generator, train_ov_generator, val_generator, test_generator = generator_init(
    img_path='data/Keras_Btz_img', batch_size=batch_size)


# # Models set up

# # # General parameters
sgd_opti = SGD(lr=0.001, momentum=0.9, nesterov=True, decay=1e-6)

# # #  Custom model
base_model = create_base_model(output=3)
base_model.compile(optimizer=sgd_opti,
                   loss='categorical_crossentropy',
                   metrics=['categorical_accuracy'])

# # #  AlexNet model

alex_model = create_alex_model()
alex_model.compile(optimizer=sgd_opti,
                   loss='categorical_crossentropy',
                   metrics=['categorical_accuracy'])

# # #  Inception v3 model

model_incep = create_incep_model(output=3, weights_cnn=None)
model_incep.compile(optimizer=sgd_opti, loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

# # # VGG16 model

model_vgg = create_vgg16_model(output=3, weights_cnn=None)
model_vgg.compile(optimizer=sgd_opti, loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])


# # #  Compute the weights if cost sensitive learning :

class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(train_generator.classes),
                                                  train_generator.classes)

class_weights = {i: class_weights[i] for i in range(3)}

# # #  Callbacks
earlystop = EarlyStopping(patience=10)
cb = TimingCallback()
schedule = StepDecay(initAlpha=0.001, factor=0.5, dropEvery=10)

callbacks = [cb, earlystop, LearningRateScheduler(schedule)]


# # # Training

# Training custom model

history = base_model.fit(
    train_ov_generator,
    epochs=100,
    validation_data=val_generator,
    validation_steps=val_generator.n//batch_size,
    steps_per_epoch=train_ov_generator.n//batch_size, # use train_generator for training set without oversampling
    callbacks=callbacks,
    # class_weight = class_weights, #uncomment for cost sensitive learning
    max_queue_size=100,
    use_multiprocessing=True
)

# Model graphs

make_train_graphs(history)

# Model save

mat_conf_base = make_conf_mat(base_model, test_gen=test_generator, print=False)
Res1 = [sum(cb.logs),
        len(history.epoch),
        mat_conf_base.ACC_Macro,
        mat_conf_base.ACC,
        mat_conf_base.F1_Macro,
        mat_conf_base.F1,
        mat_conf_base]

filehandler = open("Results/base_model_OV_res_26e_bs32", 'wb')
pickle.dump(Res1, filehandler)

base_model.save_weights("Results/base_model_OV_26e_bs_32.h5")

# Oversample + thresholding

# Training others models

# AlexNet model

cb = TimingCallback()
schedule = StepDecay(initAlpha=0.001, factor=0.5, dropEvery=10)
callbacks = [cb, earlystop, LearningRateScheduler(schedule)]

history_alex = alex_model.fit(
    train_ov_generator,
    epochs=100,
    validation_data=val_generator,
    validation_steps=val_generator.n//batch_size,
    steps_per_epoch=train_ov_generator.n//batch_size,
    callbacks=callbacks,
    # class_weight=class_weights,
    max_queue_size=100,
    use_multiprocessing=True
)

make_train_graphs(history_alex)

mat_conf_alex = make_conf_mat(alex_model, test_gen=test_generator, print=False)

Res2 = [sum(cb.logs),
        len(history_alex.epoch),
        mat_conf_alex.ACC_Macro,
        mat_conf_alex.ACC,
        mat_conf_alex.F1_Macro,
        mat_conf_alex.F1,
        mat_conf_alex]

filehandler = open("Results/alex_model_OV_27e_bs32", 'wb')
pickle.dump(Res2, filehandler)

alex_model.save_weights("Results/alex_model_OV_27e_bs32.h5")

# VGG16 model


cb = TimingCallback()
schedule = StepDecay(initAlpha=0.001, factor=0.5, dropEvery=10)
callbacks = [cb, earlystop, LearningRateScheduler(schedule)]
history_vgg = model_vgg.fit(
    train_ov_generator,
    epochs=100,
    validation_data=val_generator,
    validation_steps=val_generator.n//batch_size,
    steps_per_epoch=train_ov_generator.n//batch_size,
    callbacks=callbacks,
    # class_weight = class_weights,
    max_queue_size=100,
    use_multiprocessing=True
)

make_train_graphs(history_vgg)
make_conf_mat(model_vgg,
              test_gen=test_generator,
              print=True)

mat_conf_vgg = make_conf_mat(model_vgg, print=False, test_gen=test_generator)

Res3 = [sum(cb.logs),
        len(history_vgg.epoch),
        mat_conf_vgg.ACC_Macro,
        mat_conf_vgg.ACC,
        mat_conf_vgg.F1_Macro,
        mat_conf_vgg.F1,
        mat_conf_vgg]

filehandler = open("Results/vgg_model_OV_28e_bs32", 'wb')
pickle.dump(Res3, filehandler)

model_vgg.save_weights("Results/vgg_model_OV_28e_bs32.h5")


# Inception v3 model

cb = TimingCallback()
schedule = StepDecay(initAlpha=0.001, factor=0.5, dropEvery=10)
callbacks = [cb, earlystop, LearningRateScheduler(schedule)]

history_incep = model_incep.fit(
    train_ov_generator,
    epochs=100,
    validation_data=val_generator,
    validation_steps=val_generator.n//batch_size,
    steps_per_epoch=train_ov_generator.n//batch_size,
    callbacks=callbacks,
    # class_weight = class_weights,
    max_queue_size=100,
    use_multiprocessing=True
)

make_train_graphs(history_incep)

mat_conf_incep = make_conf_mat(model_incep, test_gen=test_generator,
                               print=False)
Res4 = [sum(cb.logs),
        len(history_incep.epoch),
        mat_conf_incep.ACC_Macro,
        mat_conf_incep.ACC,
        mat_conf_incep.F1_Macro,
        mat_conf_incep.F1,
        mat_conf_incep]


filehandler = open("Results/incep_model_OV_38e_bs32", 'wb')
pickle.dump(Res4, filehandler)

model_incep.save_weights("Results/incep_model_OV_38e_bs32.h5")


# Predict with best model Ztz

Ztz_model = create_vgg16_model(output=3, weights_cnn=None)

Ztz_model.compile(optimizer=sgd_opti,
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

Ztz_model.load_weights('Results/vgg_model_T_OV_13e_bs32.h5')
