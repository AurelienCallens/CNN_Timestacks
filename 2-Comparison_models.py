#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 25 2021

@author: Aurelien Callens

This script trains different CNN models and saves them
"""

# Importing libraries: 

from tensorflow.keras.preprocessing import image
import pandas as pd 
import numpy as np
import os 
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

#Setting seeds for reproducibility 

from numpy.random import seed
seed(1)

import tensorflow
tensorflow.random.set_random_seed(2)

# Gpu loading 

from tensorflow.keras.backend import set_session

config = tensorflow.ConfigProto()

config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.80
sess = tensorflow.Session(config=config)

set_session(sess)


################# Generators preparation #################


batch_size = 32

# data augmentation techniques : 
train_datagen = image.ImageDataGenerator(rescale=(1/255),
                                         vertical_flip=True,
                                    channel_shift_range=50.0,
                                    brightness_range=(0.5, 1.5),
                                    height_shift_range=0.2)
        
        
test_datagen = image.ImageDataGenerator(rescale=(1/255))
        
train_generator = train_datagen.flow_from_directory(
    directory="data/Keras_Btz_img/train/", # Change the directory names for your data
    color_mode="rgb",
    target_size = (224,224),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

train_ov_generator = train_datagen.flow_from_directory(
    directory="data/Keras_Btz_img/train_ov/",# Change the directory names for your data
    color_mode="rgb",
    target_size = (224,224),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)


val_generator = test_datagen.flow_from_directory(
    directory="data/Keras_Btz_img/val/", # Change the directory names for your data
    color_mode="rgb",
    target_size = (224,224),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory="data/Keras_Btz_img/test/", # Change the directory names for your data
    color_mode="rgb",
    target_size = (224,224),
    batch_size=1,
    class_mode="categorical",
    shuffle=False,
    seed=42
)

    
################# Models #################

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import VGG16
from scripts.Cnn_functions import create_base_model, create_alex_model

# Custom model

base_model = create_base_model(output = 3)
base_model.summary()
sgd_opti = SGD(lr = 0.001, momentum = 0.9, nesterov = True, decay = 1e-6)
base_model.compile(optimizer= sgd_opti, 
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
        
# AlexNet model 

alex_model = create_alex_model()
alex_model.summary()
alex_model.compile(optimizer= sgd_opti, 
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

# Inception v3 model 

incep_model = InceptionV3(include_top=False, pooling ='max',
                          weights = None, input_shape=(224, 224,3))


# Create the model
model_incep = Sequential()
model_incep.add(incep_model)
# Add new layers
model_incep.add(Flatten())
model_incep.add(Dense(512, activation='relu'))
model_incep.add(Dropout(0.5))
model_incep.add(Dense(3, activation='softmax'))

model_incep.compile(optimizer= sgd_opti, 
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

model_incep.summary()

# VGG16 model 


vgg_model = VGG16(include_top=False, pooling ='max',
                          weights = None, input_shape=(224, 224,3))

# Create the model
model_vgg = Sequential()
model_vgg.add(vgg_model)
# Add new layers
model_vgg.add(Flatten())
#model_incep.add(Dense(1024, activation='relu'))
#model_incep.add(Dropout(0.5))
model_vgg.add(Dense(512, activation='relu'))
model_vgg.add(Dropout(0.5))
model_vgg.add(Dense(3, activation='softmax'))

model_vgg.compile(optimizer= sgd_opti, 
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

model_vgg.summary()


# Compute the weights if cost sensitive learning : 
    
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(train_generator.classes), 
                train_generator.classes)

class_weights = {i : class_weights[i] for i in range(3)}


################# Callbacks for CNN training  #################

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer


# Earlystop on val loss
earlystop = EarlyStopping(patience=10)

# Learning rate decay 

class LearningRateDecay:
	def plot(self, epochs, title="Learning Rate Schedule"):
		# compute the set of learning rates for each corresponding
		# epoch
		lrs = [self(i) for i in epochs]
		# the learning rate schedule
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(epochs, lrs)
		plt.title(title)
		plt.xlabel("Epoch #")
		plt.ylabel("Learning Rate")
        
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

# Callback t orecord training time 

class TimingCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)


cb = TimingCallback()
schedule = StepDecay(initAlpha=0.001, factor=0.5, dropEvery=10)

callbacks = [cb, earlystop, LearningRateScheduler(schedule)]


################# Training #################

# Training custom model 

history = base_model.fit(  
    train_ov_generator, 
    epochs=100,
    validation_data=val_generator,
    validation_steps=val_generator.n//batch_size,
    steps_per_epoch=train_ov_generator.n//batch_size, # use train_generator for training set without oversampling
    callbacks=callbacks,
    workers = 12,
    #class_weight = class_weights, #uncomment for cost sensitive learning
    max_queue_size = 100,
    use_multiprocessing = True
)

################# Model results #################

from scripts.Make_graphs import plot_image, plot_value_array, make_train_graphs

make_train_graphs(history)

from scripts.Cnn_functions import make_conf_mat
    
make_conf_mat(base_model, test_gen = test_generator, print = True)

############## Model Saving ##############
# Baseline 


mat_conf_base = make_conf_mat(base_model, test_gen = test_generator, print = False)
Res1 = [sum(cb.logs),len(history.epoch), mat_conf_base.ACC_Macro, mat_conf_base.ACC, mat_conf_base.F1_Macro, mat_conf_base.F1, mat_conf_base] 

import pickle 
filehandler = open("Results/base_model_OV_res_26e_bs32", 'wb') 
pickle.dump(Res1, filehandler)

base_model.save_weights("Results/base_model_OV_26e_bs_32.h5")

# Oversample + thresholding

########### Training others models

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
    #class_weight = class_weights,
    workers = 11,
    max_queue_size = 100,
    use_multiprocessing = True
)

make_train_graphs(history_alex)

make_conf_mat(alex_model,
              test_gen = test_generator,
              print = True)

mat_conf_alex = make_conf_mat(alex_model,
              test_gen = test_generator,
              print = False)

Res2 = [sum(cb.logs),len(history_alex.epoch), mat_conf_alex.ACC_Macro, mat_conf_alex.ACC, mat_conf_alex.F1_Macro, mat_conf_alex.F1, mat_conf_alex] 

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
    #class_weight = class_weights,
    workers = 11,
    max_queue_size = 100,
    use_multiprocessing = True
)

make_train_graphs(history_vgg)
make_conf_mat(model_vgg,
              test_gen = test_generator,
              print = True)

mat_conf_vgg = make_conf_mat(model_vgg, print = False, test_gen = test_generator)
Res3 = [sum(cb.logs),len(history_vgg.epoch), mat_conf_vgg.ACC_Macro, mat_conf_vgg.ACC, mat_conf_vgg.F1_Macro, mat_conf_vgg.F1, mat_conf_vgg] 

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
    #class_weight = class_weights,
    workers = 11,
    max_queue_size = 100,
    use_multiprocessing = True
)

make_train_graphs(history_incep)
make_conf_mat(model_incep,test_gen = test_generator, print = True)
mat_conf_incep = make_conf_mat(model_incep, test_gen = test_generator, print = False)
Res4 = [sum(cb.logs),len(history_incep.epoch), mat_conf_incep.ACC_Macro, mat_conf_incep.ACC, mat_conf_incep.F1_Macro, mat_conf_incep.F1, mat_conf_incep] 


filehandler = open("Results/incep_model_OV_38e_bs32", 'wb') 
pickle.dump(Res4, filehandler)

model_incep.save_weights("Results/incep_model_OV_38e_bs32.h5")


#### Predict with best model Ztz


from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

Ztz_model = VGG16(include_top=False, pooling ='max',
                          weights = None, input_shape=(224, 224,3))

#for layer in incep_model.layers[:]:
#    layer.trainable = False

# Check the trainable status of the individual layers
#for layer in incep_model.layers:
#    print(layer, layer.trainable)

# Create the model
Ztz_model = Sequential()
Ztz_model.add(Ztz_model)
# Add new layers
Ztz_model.add(Flatten())
#model_incep.add(Dense(1024, activation='relu'))
#model_incep.add(Dropout(0.5))
Ztz_model.add(Dense(512, activation='relu'))
Ztz_model.add(Dropout(0.5))
Ztz_model.add(Dense(3, activation='softmax'))

sgd_opti = SGD(lr = 0.001, momentum = 0.9, nesterov = True, decay = 1e-6)
Ztz_model.compile(optimizer= sgd_opti, 
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

Ztz_model.load_weights('Results/vgg_model_T_OV_13e_bs32.h5')







