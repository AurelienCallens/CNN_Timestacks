#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 25 2021

@author: Aurelien Callens
"""

# Importing libraries: 
    
import pandas as pd 
import numpy as np
import datatable
from scripts.Build_datasets import split_resize_cnn_img

# Importing csv file containing the file path and label of the images: 
 
# Split data for Biarritz 

df_fp = pd.read_csv("./data/Biarritz_new_fp.csv")

label = df_fp.truelab
label[label  == "n"] = '0'
label[label == "i"] = '1'
label[label  == "s"] = '2'

split_resize_cnn_img(label, df_fp, root_dir ='./data/Keras_Btz_img', imb_met = "oversampling", seed_nb = 2)


# Split data for Zarautz
 
df_fp1 = pd.read_csv("./data/Zarautz_new_fp.csv")

df_fp1 = df_fp1.dropna(subset=['cnn_label'])

label1 = df_fp1['cnn_label'].apply(int).apply(str)


split_resize_cnn_img(label, df_fp1, root_dir ='./data/Keras_Ztz_img', imb_met = "oversampling", seed_nb = 2)



############################# Not mandatory !!! #############################
# Split data for Zarautz (3 smaller datasets for sensitivy analysis)

df_fp2 = pd.read_csv("./data/Zarautz_new_fp_3.csv")

df_fp2 = df_fp2.dropna(subset=['cnn_label'])

label2 = df_fp2['cnn_label'].apply(int).apply(str)


split_resize_cnn_img(label, df_fp, root_dir ='./data/Keras_Ztz_img_3', imb_met = "oversampling", seed_nb = 2)



train1s = df_fp2[df_fp2['cnn_label']== 0].sample(frac = 0.333)
train2s = df_fp2[df_fp2['cnn_label']== 0].drop(train1s.index).sample(frac = 0.5)
train3s = df_fp2[df_fp2['cnn_label']== 0].drop(train1s.index).drop(train2s.index)

train1i = df_fp2[df_fp2['cnn_label']== 1].sample(frac = 0.333)
train2i = df_fp2[df_fp2['cnn_label']== 1].drop(train1i.index).sample(frac = 0.5)
train3i = df_fp2[df_fp2['cnn_label']== 1].drop(train1i.index).drop(train2i.index)

train1o = df_fp2[df_fp2['cnn_label']== 2].sample(frac = 0.333)
train2o = df_fp2[df_fp2['cnn_label']== 2].drop(train1o.index).sample(frac = 0.5)
train3o = df_fp2[df_fp2['cnn_label']== 2].drop(train1o.index).drop(train2o.index)

data_1 = pd.concat([train1s, train1i,train1o])
data_2 = pd.concat([train2s, train2i,train2o])
data_3 = pd.concat([train3s, train3i,train3o])


label_1 = data_1['cnn_label'].apply(int).apply(str)
label_2 = data_2['cnn_label'].apply(int).apply(str)
label_3 = data_3['cnn_label'].apply(int).apply(str)

split_resize_cnn_img(label_1, data_1, root_dir ='./data/Keras_Ztz_img_3_sens1', imb_met = "oversampling", seed_nb = 2)
split_resize_cnn_img(label_2, data_2, root_dir ='./data/Keras_Ztz_img_3_sens2', imb_met = "oversampling", seed_nb = 2)
split_resize_cnn_img(label_3, data_3, root_dir ='./data/Keras_Ztz_img_3_sens3', imb_met = "oversampling", seed_nb = 2)



