

#Image Classfication pipeline 

from imutils import paths 
import cv2
import keras
from keras.preprocessing.image import img_to_array
import os
import numpy as np 
import time 
from sklearn.model_selection import train_test_split


train_data = []
train_label = []

print("empty set...",len(train_data))

start = time.time()


dataPathsmelanoma =sorted(list(paths.list_images(".../train/train/melanoma")))

for x in dataPathsmelanoma: 
    image = cv2.imread(x)
    image = cv2.resize(image, (32,32))
    image = img_to_array(image)
    
    train_data.append(image)
    
    label = os.path.basename((x.split(os.path.sep)[-2]))
    
    if label == "melanoma":
        label = 0
    elif label == "nevus":
        label = 1
    else:
        label = 2
    
    train_label.append(label)
    
print("first set...",len(train_data))


dataPathsnevus =sorted(list(paths.list_images(".../train/train/nevus")))

for x in dataPathsnevus: 
    image = cv2.imread(x)
    image = cv2.resize(image, (32,32))
    image = img_to_array(image)
    
    train_data.append(image)
    
    label = os.path.basename((x.split(os.path.sep)[-2]))
    
    if label == "melanoma":
        label = 0
    elif label == "nevus":
        label = 1
    else:
        label = 2
    
    train_label.append(label)
    
print("second set...",len(train_data))

dataPathsseborrheic_keratosis =sorted(list(paths.list_images("..../train/seborrheic_keratosis")))

for x in dataPathsseborrheic_keratosis: 
    image = cv2.imread(x)
    image = cv2.resize(image, (32,32))
    image = img_to_array(image)
    
    train_data.append(image)
    
    label = os.path.basename((x.split(os.path.sep)[-2]))
    
    if label == "melanoma":
        label = 0
    elif label == "nevus":
        label = 1
    else:
        label = 2
    
    
    train_label.append(label)
    
print("third set...",len(train_data))
    
train_data = np.array(train_data, dtype = "float")/255.0
        
train_label = np.array(train_label)  

#print(len(train_label)) 

end = time.time() - start

print("minutes: ",end/60)

#print(train_label) 

(train_X, test_X, train_y, test_y) = train_test_split(train_data, train_label, test_size = 0.25, random_state =41)


num_classes = 3 

train_y = keras.utils.to_categorical(train_y, num_classes)

test_y = keras.utils.to_categorical(test_y, num_classes)


print("train y train length", len(train_y))

print("train y test length", len(test_y))

print("x_train shape", train_X.shape)



# Model Architecture

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation 

model = Sequential()

model.add(Conv2D(filters=4, kernel_size=2, padding='same', activation='relu', input_shape=(32, 32, 3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())

model.add(Dense(2, activation='relu'))

model.add(Dense(3, activation='softmax'))

model.summary()



# compile model

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ['accuracy'])


from keras.callbacks import ModelCheckpoint

# train model & save weights 

checkpointer = ModelCheckpoint(filepath = '..../save/model.weights.best.hdf5',verbose =1 ,save_best_only =True)

version = model.fit(train_X, train_y, batch_size =32, epochs = 3, validation_data = (test_X, test_y), callbacks =[checkpointer], verbose=2, shuffle=True)

    
