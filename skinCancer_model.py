# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 23:12:42 2018

@author: D
"""
""" This only inlcuded melanoma file """
#Image Classfication pipeline 

import argparse
from imutils import paths 
import cv2
from keras.preprocessing.image import img_to_array
import os
import numpy as np 
from sklearn.model_selection import train_test_split

obj = argparse.ArgumentParser()

obj.add_argument("-d", "--dataset", required = True, help ="path")

obj.add_argument("-s", "--saved", required = True, help= "saved location path")

args = vars(obj.parse_args())

train_data = []
train_label = []

dataPaths = sorted(list(paths.list_images(args["dataset"])))

print(args)

for x in dataPaths:
    print(x)
    image = cv2.imread(x)
    image = cv2.resize(image, (32,32))
    image = img_to_array(image)
    
    train_data.append(image)
    
    label = x.split(os.path.sep)[-2]
    
    if label == "melanoma":
        train_label  = 1
        train_label.append(label)
    elif label =="nevus":
        train_label = 2
        train_label.append(label)
    else: 
        train_label = 3
        train_label.append(label)
        
train_data = np.array(train_data, dtype = "float")/255.0
        
train_label = np.array(train_label)    

(train_X, test_X, train_y, test_y) = train_test_split(train_data, train_label, test_size = 0.25, random_state =41)
    