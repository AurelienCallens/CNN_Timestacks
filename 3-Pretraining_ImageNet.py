#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 25 2021

@author: Aurelien Callens
"""

import pickle
import numpy as np

from sklearn.utils import class_weight
from numpy.random import seed
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
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
#  Imagenet weights
model_incep = create_incep_model(output=3, weights_cnn='imagenet')
model_incep.compile(optimizer=sgd_opti, loss='categorical_crossentropy',
                    metrics=['categorical_accuracy'])

# # # VGG16 model
#  Imagenet weights
model_vgg = create_vgg16_model(output=3, weights_cnn='imagenet')
model_vgg.compile(optimizer=sgd_opti, loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])


# # #  Callbacks
earlystop = EarlyStopping(patience=10)
cb = TimingCallback()
schedule = StepDecay(initAlpha=0.001, factor=0.5, dropEvery=10)

callbacks = [cb, earlystop, LearningRateScheduler(schedule)]



# # # Model training and saving

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

Res5 = [sum(cb.logs),
        len(history_vgg.epoch),
        mat_conf_vgg.ACC_Macro,
        mat_conf_vgg.ACC,
        mat_conf_vgg.F1_Macro,
        mat_conf_vgg.F1,
        mat_conf_vgg]

filehandler = open("Results_Btz/vgg_model_T_OV_34e_bs32", 'wb')
pickle.dump(Res5, filehandler)
model_vgg.save_weights("Results_Btz/vgg_model_T_OV_34e_bs32.h5")


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
Res6 = [sum(cb.logs),
        len(history_incep.epoch),
        mat_conf_incep.ACC_Macro,
        mat_conf_incep.ACC,
        mat_conf_incep.F1_Macro,
        mat_conf_incep.F1,
        mat_conf_incep]


filehandler = open("Results/incep_model_T_OV_21e_bs32", 'wb')
pickle.dump(Res6, filehandler)

model_incep.save_weights("Results/incep_model_T_OV_21e_bs32.h5")
