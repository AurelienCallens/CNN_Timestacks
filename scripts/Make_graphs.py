#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 14:28:29 2020

@author: aurelien
"""


from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 




def make_train_graphs(history):
    """
    Plot the loss during the training for the training and validation sets.
    
    :param history: Fitted keras model  
    """
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.plot(history.history['loss'], color='b', label="Training loss")
    ax1.plot(history.history['val_loss'], color='r', label="validation loss")
    ax1.set_xticks(np.arange(1, history.epoch[-1], 1))
    #ax1.set_yticks(np.arange(0, 1, 0.1))
    
    ax2.plot(history.history['categorical_accuracy'], color='b', label="Training accuracy")
    ax2.plot(history.history['val_categorical_accuracy'], color='r',label="Validation accuracy")
    ax2.set_xticks(np.arange(1, history.epoch[-1], 1))
    
    legend = plt.legend(loc='best', shadow=True)
    plt.tight_layout()
    plt.show()





label_y = ["Swash", "Collision", "Overwash"]

def plot_image(i, predictions_array, y_true, fp_img, wall = True):
    """
    Plot the i th timestack and label the plot this way : Predicted label (True label).
    The color of the label on the correctness of the prediction (green for 
    good prediction and red for bad prediction)
    
    :param i: Index of te timestack image 
    :param predictions_array: Array with CNN predictions   
    :param y_true: True label   
    :param fp_img: Filepath of the images  
    :param wall: If True : represent the defense seawall by two lines. The coordinates correspond to the sea wall of Biarritz. They must be changed if used on other sites.  
    """
    predictions_array, y_true = predictions_array[i], y_true[i] ,
    img = Image.open(fp_img[i])
  
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)
  
    if wall:
        plt.axvline(x=174, color = 'r', linewidth=0.9)
        plt.axvline(x=177.2, color = 'r', linewidth=0.9)
  
    predicted_label = np.argmax(predictions_array)
    true_label = y_true
  
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(label_y[predicted_label],
                                100*np.max(predictions_array),
                                label_y[true_label]),
                                color=color)




def plot_value_array(i, predictions_array, true_label):
       """
    Plot the probabilities of belonging to the 3 classe of the i th timestack. 
    The bar of the true label is blue if the prediction is correct and red otherwise.
    
    :param i: Index of te timestack image 
    :param predictions_array: Array with CNN predictions   
    :param y_true: True label   
    """
    
    
    
    
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks(range(3))
  plt.yticks([])
  thisplot = plt.bar(range(3), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  plt.xticks(range(3), label_y, rotation=45)
  
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
 

