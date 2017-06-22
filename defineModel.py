# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 14:47:13 2017

@author: John
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers import MaxPooling2D

def getModel():
    convDropoutProb = 0.5
    fcnDropoutProb = 0.5
    
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3), dim_ordering='tf'))
    
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), init='normal', border_mode='valid', dim_ordering='tf', activation='relu'))
    model.add(Dropout(convDropoutProb))
    
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), init='normal', border_mode='valid', dim_ordering='tf', activation='relu'))
    model.add(Dropout(convDropoutProb))
    
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), init='normal', border_mode='valid', dim_ordering='tf', activation='relu'))
    model.add(Dropout(convDropoutProb))
    
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), init='normal', activation='relu'))
    
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), init='normal', activation='relu'))
    
    model.add(Flatten())
    model.add(Dropout(fcnDropoutProb))
    model.add(Dense(1164, init='normal', activation='relu'))
    model.add(Dropout(fcnDropoutProb))
    model.add(Dense(100, init='normal', activation='relu'))
    model.add(Dropout(fcnDropoutProb))
    model.add(Dense(50, init='normal', activation='relu'))
    model.add(Dropout(fcnDropoutProb))
    model.add(Dense(10, init='normal', activation='relu'))
    model.add(Dropout(fcnDropoutProb))
    model.add(Dense(1))
    return model