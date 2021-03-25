#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 25 2021

@author: Aurelien Callens
"""

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


################# Models #################


from tensorflow.keras.optimizers import SGD, Adam
from scripts.Cnn_functions import create_base_model, create_alex_model


sgd_opti = SGD(lr = 0.0001, momentum = 0.9, nesterov = True, decay = 1e-6)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3

# 
incep_model = InceptionV3(include_top=False, pooling ='max',
                          weights = 'imagenet', input_shape=(224, 224,3)) # weights = 'imagenet' for pretraining



# Create the model
model_incep = Sequential()
model_incep.add(incep_model) # Add the inception model with Imagenet weights 
# Add new layers
model_incep.add(Flatten())
#model_incep.add(Dense(1024, activation='relu'))
#model_incep.add(Dropout(0.5))
model_incep.add(Dense(512, activation='relu'))
model_incep.add(Dropout(0.5))
model_incep.add(Dense(3, activation='softmax'))

model_incep.compile(optimizer= sgd_opti, 
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

model_incep.summary()


from tensorflow.keras.applications import VGG16


vgg_model = VGG16(include_top=False, pooling ='max',
                          weights = 'imagenet', input_shape=(224, 224,3))# weights = 'imagenet' for pretraining



# Create the model
model_vgg = Sequential()
model_vgg.add(vgg_model)# Add the vgg16 model with Imagenet weights 
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


#### Model training 

from scripts.Make_graphs import plot_image, plot_value_array, make_train_graphs



from pycm import *
def make_conf_mat(model, threshold= False, print = True):
    predict = model.predict(test_generator, steps = test_generator.n)
    #predict the test set 
    #preds=model.predict_generator(generator = test_datagen.flow(X_test, y_test, batch_size = len(y_test), shuffle = False),
    #                              steps = 1, verbose = 2)
    if(threshold):
        prob = pd.value_counts(train_generator.classes)/train_generator.n
        predict = pd.DataFrame(predict)
        predict = predict/prob
        predict = (predict.T / predict.T.sum()).T
        predict = np.array(predict)
    
    y_classes = predict.argmax(axis=-1)
    #K.clear_session()    
    cm = ConfusionMatrix(actual_vector=test_generator.classes.tolist(), predict_vector=y_classes.tolist())
    #print(cm)
    if(print):
        cm.print_matrix()
        cm.stat(summary = True)
    else:
        return(cm)
    

############## Model Saving ##############


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
make_conf_mat(model_vgg, print = True, threshold = False)

mat_conf_vgg = make_conf_mat(model_vgg, print = False, threshold = False)
Res3 = [sum(cb.logs),len(history_vgg.epoch), mat_conf_vgg.ACC_Macro, mat_conf_vgg.ACC, mat_conf_vgg.F1_Macro, mat_conf_vgg.F1, mat_conf_vgg] 

import pickle 
filehandler = open("Results_Btz/vgg_model_T_OV_34e_bs32", 'wb') 
pickle.dump(Res3, filehandler)

model_vgg.save_weights("Results_Btz/vgg_model_T_OV_34e_bs32.h5")




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
make_conf_mat(model_incep, print = True, threshold = False)
mat_conf_incep = make_conf_mat(model_incep, print = False, threshold = False)
Res4 = [sum(cb.logs),len(history_incep.epoch), mat_conf_incep.ACC_Macro, mat_conf_incep.ACC, mat_conf_incep.F1_Macro, mat_conf_incep.F1, mat_conf_incep] 


filehandler = open("Results/incep_model_T_OV_21e_bs32", 'wb') 
pickle.dump(Res4, filehandler)

model_incep.save_weights("Results/incep_model_T_OV_21e_bs32.h5")
