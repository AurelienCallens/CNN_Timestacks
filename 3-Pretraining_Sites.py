#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25  2021

@author: Aurelien Callens
"""


import pickle
import numpy as np

from sklearn.utils import class_weight
from numpy.random import seed
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.preprocessing import image
from scripts.Cnn_functions import *
from sklearn.utils import class_weight
from scripts.Make_graphs import make_train_graphs
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

# Analysis
# # Generator initialization
batch_size = 32
train_generator, train_ov_generator, val_generator, test_generator = generator_init(
    img_path='data/Keras_Btz_img', batch_size=batch_size)

sgd_opti = SGD(lr=0.001, momentum=0.9, nesterov=True, decay=1e-6)

# Best model Zarautz

Ztz_model = create_vgg16_model(output=3, weights_cnn=None)
Ztz_model.compile(optimizer=sgd_opti,
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

# Load the weight of the best model for Zarautz :

Ztz_model.load_weights('Weights_best_models/Zarautz/2-Pretraining_Imagenet/vgg_model_T_OV_13e_bs32.h5')

# # #  Callbacks
earlystop = EarlyStopping(patience=10)
cb = TimingCallback()
schedule = StepDecay(initAlpha=0.001, factor=0.5, dropEvery=10)

callbacks = [cb, earlystop, LearningRateScheduler(schedule)]

# Evaluate the performance of the best CNN of Zarautz on Biarritz data :

make_conf_mat(Ztz_model, test_gen=test_generator, print=True)


# Train the model with pretrained weights:

history = Ztz_model.fit(
    train_ov_generator,
    epochs=100,
    validation_data=val_generator,
    validation_steps=val_generator.n//batch_size,
    steps_per_epoch=train_ov_generator.n//batch_size,
    callbacks=callbacks,
    max_queue_size=100,
    use_multiprocessing=True)

# Reavaluate the performances :

make_conf_mat(Ztz_model, test_gen=test_generator, print=True)


mat_conf_tr = make_conf_mat(Ztz_model, test_gen=test_generator, print=False)

Res7 = [sum(cb.logs),
        len(history.epoch),
        mat_conf_tr.ACC_Macro,
        mat_conf_tr.ACC,
        mat_conf_tr.F1_Macro,
        mat_conf_tr.F1,
        mat_conf_tr]

# Save the model :

filehandler = open("Res_transfer/vgg_model_Ztz_Btz_OV_19e_bs32", 'wb')
pickle.dump(Res7, filehandler)

Ztz_model.save_weights("Res_transfer/vgg_model_Ztz_Btz_OV_19e_bs32.h5")
