#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 25 2021

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


#x, y = train_generator.next()
#for i in range(0,4):
#    image_t = x[i]
#    plt.imshow(image_t)
#    plt.show()
    
################# Models #################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import SGD, Adam

Btz_vgg_model = VGG16(include_top=False, pooling ='max',
                          weights = None, input_shape=(224, 224,3))

#for layer in incep_model.layers[:]:
#    layer.trainable = False

# Check the trainable status of the individual layers
#for layer in incep_model.layers:
#    print(layer, layer.trainable)

# Create the model
Btz_model = Sequential()
Btz_model.add(Btz_vgg_model)
# Add new layers
Btz_model.add(Flatten())
#model_incep.add(Dense(1024, activation='relu'))
#model_incep.add(Dropout(0.5))
Btz_model.add(Dense(512, activation='relu'))
Btz_model.add(Dropout(0.5))
Btz_model.add(Dense(3, activation='softmax'))

sgd_opti = SGD(lr = 0.001, momentum = 0.9, nesterov = True, decay = 1e-6)
Btz_model.compile(optimizer= sgd_opti, 
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

Btz_model.load_weights('Weights_best_models/Biarritz/1-Pretraining_Imagnet/vgg_model_T_OV_20e_bs32.h5')



################# Model results #################

from scripts.Cnn_functions import make_conf_mat

make_conf_mat(Btz_model,
              test_gen = test_generator,
              print = True)



################# Error visualisation #################

from scripts.Make_graphs import plot_image, plot_value_array

predict = Btz_model.predict(test_generator, steps = test_generator.n)
y_classes = predict.argmax(axis=-1)

deca = 18
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(4*2*num_cols, 4*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i+deca, predict, test_generator.classes, test_generator.filepaths, wall = False)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i+deca, predict, test_generator.classes)
plt.tight_layout()
#plt.subplots_adjust(wspace=0.05, hspace=0.2, left = 0.5, bottom = 0.05, top = 0.5)
plt.show()


# We want to see only the errors: 

errors_ind = np.where(y_classes != np.array(test_generator.classes))[0]
len(errors_ind)

test_generator.classes[errors_ind]

num_rows = 4
num_cols = 4
num_images = num_rows*num_cols
plt.figure(figsize=(4*2*num_cols, 4*num_rows))
for i in range(num_images):
    ind_img = errors_ind[i]
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(ind_img, predict, test_generator.classes, test_generator.filepaths, wall = True)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(ind_img, predict, test_generator.classes)
plt.tight_layout()
#plt.subplots_adjust(wspace=0.05, hspace=0.2, left = 0.5, bottom = 0.05, top = 0.5)
plt.show()
