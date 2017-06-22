# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 15:08:32 2017

@author: John
"""

import cv2
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Lambda
from keras.layers import Conv2D, Cropping2D
from keras.layers import MaxPooling2D
from sklearn.utils import shuffle
from keras.models import load_model
import defineModel

# Set hyperparameter values
isTransferLearning = True
batch_size = 128
epochs = 10

samples = []
if isTransferLearning:
    dataFile = r'./newData/driving_log.csv' 
    path = r'./newData/IMG/'
else:
    dataFile = r'./data/driving_log.csv' 
    path = r'./data/IMG/'
    
with open(dataFile) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Allow for transfer learning
def doTransferLearning():
    model = defineModel.getModel()
    model.load_weights(r'./best_model.h5')
    for layer in model.layers[:-9]:
        layer.trainable = False
    return model

# Get center, left, and rights images along with their flipped variants
def generator(samples, batch_size=32):
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            car_images = []
            steering_angles = []
            for batch_sample in batch_samples:
                img_center = cv2.imread(path+batch_sample[0].split('\\')[-1])
                img_left   = cv2.imread(path+batch_sample[1].split('\\')[-1])
                img_right  = cv2.imread(path+batch_sample[2].split('\\')[-1])
                
                correction = 0.3 # this is a parameter to tune
                steering_center = float(batch_sample[3])
                steering_left   = steering_center + correction
                steering_right  = steering_center - correction
                
                # Augment with flipped images
                center_flipped = np.fliplr(img_center)
                left_flipped = np.fliplr(img_left)
                right_flipped = np.fliplr(img_right)
                steering_center_flipped = -steering_center
                steering_left_flipped = -steering_left
                steering_right_flipped = -steering_right
                
                # add images and angles to data set
                car_images.extend([img_center, img_left, img_right, center_flipped, left_flipped, right_flipped])
                steering_angles.extend([steering_center, steering_left, steering_right, 
                                        steering_center_flipped, steering_left_flipped, steering_right_flipped])
                
            # trim image to only see section with road
            X_train = np.array(car_images)
            y_train = np.array(steering_angles)
            yield shuffle(X_train, y_train)

# Get center image only
def generator2(samples, batch_size=32):
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            car_images = []
            steering_angles = []
            for batch_sample in batch_samples:
                img_center = cv2.imread(path+batch_sample[0].split('\\')[-1])
                steering_center = float(batch_sample[3])
                
                # add images and angles to data set
                car_images.extend([img_center])
                steering_angles.extend([steering_center])
                
            # trim image to only see section with road
            X_train = np.array(car_images)
            y_train = np.array(steering_angles)
            yield shuffle(X_train, y_train)

# Get center, left, and right images            
def generator3(samples, batch_size=32):
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            car_images = []
            steering_angles = []
            for batch_sample in batch_samples:
                img_center = cv2.imread(path+batch_sample[0].split('\\')[-1])
                img_left   = cv2.imread(path+batch_sample[1].split('\\')[-1])
                img_right  = cv2.imread(path+batch_sample[2].split('\\')[-1])
                
                correction = 0.3 # this is a parameter to tune
                steering_center = float(batch_sample[3])
                steering_left   = steering_center + correction
                steering_right  = steering_center - correction
                
                # add images and angles to data set
                car_images.extend([img_center, img_left, img_right])
                steering_angles.extend([steering_center, steering_left, steering_right])
                
            # trim image to only see section with road
            X_train = np.array(car_images)
            y_train = np.array(steering_angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
factor = 6   # 6 for original generator, 1 for generator2, 3 for generator3
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

# Train and save model
print('Training model...')
if isTransferLearning:
    model = doTransferLearning()
else:
    model = defineModel.getModel()
model.compile(optimizer='adam', loss='mse')
print(model.summary())

model.fit_generator(train_generator, samples_per_epoch=factor*len(train_samples),
             validation_data=validation_generator,
            nb_val_samples=len(validation_samples), nb_epoch=epochs)
print('Training complete! Saving model...')
model.save_weights('model.h5') 
