#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25  2021

@author: Aurelien Callens
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

train_datagen = image.ImageDataGenerator(rescale=(1/255),
                                         vertical_flip=True,
                                    channel_shift_range=50.0,
                                    brightness_range=(0.5, 1.5),
                                    height_shift_range=0.2)
        
        
test_datagen = image.ImageDataGenerator(rescale=(1/255))
        
train_generator = train_datagen.flow_from_directory(
    directory="data/Keras_Btz_img/train/",
    color_mode="rgb",
    target_size = (224,224),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

train_ov_generator = train_datagen.flow_from_directory(
    directory="data/Keras_Btz_img/train_ov/",
    color_mode="rgb",
    target_size = (224,224),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)


val_generator = test_datagen.flow_from_directory(
    directory="data/Keras_Btz_img/val/",
    color_mode="rgb",
    target_size = (224,224),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory="data/Keras_Btz_img/test/",
    color_mode="rgb",
    target_size = (224,224),
    batch_size=1,
    class_mode="categorical",
    shuffle=False,
    seed=42
)


    
############## Best model Ztz #################

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16

########### Create same architecture : 
    
Ztz_vgg_model = VGG16(include_top=False, pooling ='max',
                          weights = None, input_shape=(224, 224,3))

# Create the model
Ztz_model = Sequential()
Ztz_model.add(Ztz_vgg_model)
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

#### Load the weight of the best model for Zarautz : 
    
Ztz_model.load_weights('Weights_best_models/Zarautz/2-Pretraining_Imagenet/vgg_model_T_OV_13e_bs32.h5')

################# Callback #################

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

earlystop = EarlyStopping(patience=10)

import matplotlib.pyplot as plt
import numpy as np
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


from timeit import default_timer as timer

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

### Evaluate the performance of the best CNN of Zarautz on Biarritz data : 

from scripts.Cnn_functions import make_conf_mat
    
make_conf_mat(Ztz_model, test_gen = test_generator, print = True)


### Train the model with pretrained weights : 
    
history = Ztz_model.fit(  
    train_ov_generator, 
    epochs=100,
    validation_data=val_generator,
    validation_steps=val_generator.n//batch_size,
    steps_per_epoch=train_ov_generator.n//batch_size,
    callbacks=callbacks,
    workers = 12,
    max_queue_size = 100,
    use_multiprocessing = True
)

### Reavaluate the performances : 

make_conf_mat(Ztz_model, test_gen = test_generator, print = True)


mat_conf_tr = make_conf_mat(Ztz_model, test_gen = test_generator, print = False)

Res4 = [sum(cb.logs),len(history.epoch), mat_conf_tr.ACC_Macro, mat_conf_tr.ACC, mat_conf_tr.F1_Macro, mat_conf_tr.F1,
        mat_conf_tr] 

# Save the model : 

import pickle 
filehandler = open("Res_transfer/vgg_model_Ztz_Btz_OV_19e_bs32", 'wb') 
pickle.dump(Res4, filehandler)

Ztz_model.save_weights("Res_transfer/vgg_model_Ztz_Btz_OV_19e_bs32.h5")