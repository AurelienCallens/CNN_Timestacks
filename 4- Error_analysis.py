#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 25 2021

@author: Aurelien Callens

"""

import numpy as np

from tensorflow.keras.optimizers import SGD
from scripts.Cnn_functions import *
from scripts.Make_graphs import plot_image, plot_value_array
from scripts.Cnn_functions import make_conf_mat
import matplotlib.pyplot as plt


# Analysis

batch_size = 32
train_generator, train_ov_generator, val_generator, test_generator = generator_init(
    img_path='data/Keras_Btz_img', batch_size=batch_size)


sgd_opti = SGD(lr=0.001, momentum=0.9, nesterov=True, decay=1e-6)

Btz_model = create_vgg16_model(output=3, weights_cnn=None)
Btz_model.compile(optimizer= sgd_opti,
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

Btz_model.load_weights('Weights_best_models/Biarritz/1-Pretraining_Imagnet/vgg_model_T_OV_20e_bs32.h5')

# Model results


make_conf_mat(Btz_model, test_gen=test_generator, print=True)



################# Error visualisation #################


predict = Btz_model.predict(test_generator, steps=test_generator.n)
y_classes = predict.argmax(axis=-1)

deca = 18
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(4*2*num_cols, 4*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i+deca, predict, test_generator.classes,
               test_generator.filepaths, wall=False)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i+deca, predict, test_generator.classes)
plt.tight_layout()
# plt.subplots_adjust(wspace=0.05, hspace=0.2, left=0.5, bottom=0.05, top=0.5)
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
    plot_image(ind_img, predict, test_generator.classes,
               test_generator.filepaths, wall=True)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(ind_img, predict, test_generator.classes)
plt.tight_layout()
# plt.subplots_adjust(wspace=0.05, hspace=0.2, left = 0.5, bottom = 0.05, top = 0.5)
plt.show()
