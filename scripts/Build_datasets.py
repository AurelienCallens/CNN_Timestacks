#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:14:28 2020

@author: Aurelien Callens

This script contains a function which prepares the images to be used by Keras
generators by splitting them into Train/val/test and resizing them
"""

import os
import shutil
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def safe_copy(file_path, out_dir, dst=None):
    """Safely copy a file to the specified directory. If a file with the same
    name already exists, the copied file name is altered to preserve both.
    Source : Stack Overflow https://stackoverflow.com/questions/33282647/python-shutil-copy-if-i-have-a-duplicate-file-will-it-copy-to-new-location
    :param str file_path: Path to the file to copy.
    :param str out_dir: Directory to copy the file into.
    :param str dst: New name for the copied file. If None, use the name of
    the original file.
    """
    name = dst or os.path.basename(file_path)
    if not os.path.exists(os.path.join(out_dir, name)):
        shutil.copy(file_path, os.path.join(out_dir, name))
    else:
        base, extension = os.path.splitext(name)
        i = 1
        while os.path.exists(
                os.path.join(out_dir, '{}_{}{}'.format(base, i, extension))):
            i += 1
        shutil.copy(file_path,
                    os.path.join(out_dir, '{}_{}{}'.format(base, i,
                                                           extension)))


def split_resize_cnn_img(img_labels, img_fp, root_dir, imb_met, seed_nb=1,
                         test_ratio=0.2, val_ratio=0.2, img_size=[224, 224]):
    """
    This function serves to split the images into the train/val/test
    repositories. It prepares the images for the flow_from_directory function
    of keras, so inside the images are stored inside subrepositories
    corresponding to their class. In addition to spliting the images and
    copying them into the right repositories, this function also resize the
    images and allow the user to oversample the minority classes in case of
    imbalanced dataset.

    :param list img_labels: Labels of the images
    :param str img_fp: Paths of the images
    :param str root_dir: Directory where the Train/val/test repositories will
    be created.
    :param str imb_met: None or "oversampling". None if the class ratio should
    be conserved and "oversampling" to create balance dataset with oversampling
    :param int seed_nb: Set a seed for reproducible splits
    :param int test_ratio: Proportion of the test set
    :param int val_ratio: Proportion of the validation set in the training set
    :param int img_size: Size of the image after resizing
    """

    # Printing number of classes for verificatio
    classes_dir = np.unique(img_labels)
    print('Number of class: {}, with: {}'.format(len(classes_dir), classes_dir))
    print('\n')

    # Splitting the dataset in train/val/test with given ratio +
    # setting seeds for reproducibility:
    np.random.seed(seed_nb)

    # First split between train and test
    x_trai, x_test, y_trai, y_test = train_test_split(img_fp, img_labels,
                                                      test_size=test_ratio)

    # Second split within train for validation
    x_train, x_val, y_train, y_val = train_test_split(
        img_fp.loc[y_trai.index, ], img_labels[y_trai.index],
        test_size=val_ratio)

    # Printing to verify the splits
    print('Overall data')
    print(img_labels.value_counts())
    print('Split for train:')
    print(y_train.value_counts())
    print('Split for val:')
    print(y_val.value_counts())
    print('Split for test:')
    print(y_test.value_counts())

    # Create train/val/test directory and subfolder associated with class for
    # keras flow_from_directory function and copy the resized images inside

    # Baseline method:

    for cls in classes_dir:
        os.makedirs(root_dir+'/train/class' + cls, exist_ok=True)
        train_FileNames = x_train.new_fp.loc[y_train == cls]
        for name in train_FileNames:
            img = Image.open(name)
            new_img = img.resize(img_size, Image.ANTIALIAS)
            new_img.save(fp=root_dir + '/train/class' + cls + '/' +
                         os.path.basename(name))

        os.makedirs(root_dir + '/val/class' + cls, exist_ok=True)
        val_FileNames = x_val.new_fp.loc[y_val == cls]
        for name in val_FileNames:
            img = Image.open(name)
            new_img = img.resize(img_size, Image.ANTIALIAS)
            new_img.save(fp=root_dir + '/val/class' + cls + '/' +
                         os.path.basename(name))

        os.makedirs(root_dir + '/test/class' + cls, exist_ok=True)
        test_FileNames = x_test.new_fp.loc[y_test == cls]
        for name in test_FileNames:
            img = Image.open(name)
            new_img = img.resize(img_size, Image.ANTIALIAS)
            new_img.save(fp=root_dir + '/test/class' + cls + '/' +
                         os.path.basename(name))

    # Oversampling method for training split only:
    if imb_met == "oversampling":
        maj_class = np.unique(y_train)[np.argmax(
            np.array(y_train.value_counts()))]
        eff_maj_class = y_train.value_counts()[maj_class]

        for cls in classes_dir:
            # Copy the directory of the majority class
            if cls == maj_class:
                shutil.copytree(root_dir + '/train/class' + cls + '/',
                                dst=root_dir + '/train_ov/class' + cls + '/')

            # If minority class perform oversampling to completely balance the
            # training set
            else:
                os.makedirs(root_dir + '/train_ov/class' + cls, exist_ok=True)

                # Find how many images we have to duplicate for complete balance
                diff = eff_maj_class - y_train.value_counts()[cls]

                # Get the path of the resized images of the minority class
                train_FileNames = []
                for i in x_train.new_fp.loc[y_train == cls]:
                    train_FileNames.append(root_dir + '/train/class' + cls +
                                           '/' + os.path.basename(i))

                # Oversample the images to balance the dataset
                train_FileNames.extend(random.choices(train_FileNames,
                                                      k=diff))

                # Copy the images and their duplicate in  /train_ov/ directory
                for i in range(len(train_FileNames)):
                    safe_copy(file_path=train_FileNames[i],
                              out_dir=root_dir + '/train_ov/class' + cls + '/')
