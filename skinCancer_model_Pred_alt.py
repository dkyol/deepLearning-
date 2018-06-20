# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 19:11:04 2018

@author: D
"""
from imutils import paths
import cv2
import numpy as np  
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os 
import pandas as pd  
 
#import and preprocess X_test data 

test_data = []
label_data = []

#get melanoma test data 

data_paths_melanoma = sorted(list(paths.list_images("/dermatologist-ai/data/test/melanoma")))

for x in data_paths_melanoma:
    image = cv2.imread(x)
    image = cv2.resize(image, (32,32))
    image = img_to_array(image)
    
    test_data.append(image)
    
    path = os.path.basename(x)
    
    label_data.append(path)

data_paths_nevus = sorted(list(paths.list_images("dermatologist-ai/data/test/nevus")))

for x in data_paths_nevus:
    image = cv2.imread(x)
    image =cv2.resize(image,(32,32))
    image = img_to_array(image)
    
    test_data.append(image)
    
    path = os.path.basename(x)
    
    label_data.append(path)
    
data_paths_Pathsseborrheic = sorted(list(paths.list_images("DeepLearningCancerDetection/dermatologist-ai/data/test/seborrheic_keratosis")))

for x in data_paths_Pathsseborrheic:
    image = cv2.imread(x)
    image = cv2.resize(image, (32,32))
    image = img_to_array(image)
    
    test_data.append(image)
    
    path = os.path.basename(x)
    
    label_data.append(path)
    
print("the number of loaded test points", len(test_data))

# convert to array, transform 0 to 1 

test_data = np.array(test_data, dtype ="float")/255

# load model weights

model = load_model("DeepLearningCancerDetection/project/Model_Weights/model.weights.best.hdf5")

model_predictions = model.predict(test_data)

results_label = pd.DataFrame(data = label_data, columns=['Id'])

results_data = pd.DataFrame(data = model_predictions[:,:-1], columns =['task_1', 'task_2'])

results = results_label.join(results_data, how = 'outer')

results.to_csv('DeepLearningCancerDetection/project/csv_results/results.csv')

