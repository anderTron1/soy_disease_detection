#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 10:28:10 2021

@author: andre
"""

import warnings
warnings.filterwarnings('ignore')

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, BatchNormalization
from keras import regularizers
from keras import backend as K
import keras

class Architectures:
    @staticmethod
    # LiSHT
    # activation=LiSHT
    def LiSHT(x):
        return x * K.tanh(x)
    
    def architecture_train_model(class_names, drop, l2):
        # Creating the model using the Sequential API
        print("[INFO] network architecture...")
        model = Sequential()
        #input_shape = train_images[0].shape
        model.add(Conv2D(filters=64, kernel_size=5, strides=1, padding="same", activation="relu"))#, input_shape = input_shape))(128,128,3)))
        model.add(MaxPool2D(pool_size=2))
        model.add(Dropout(drop))
        model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=2))
        model.add(Dropout(drop))
        model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=2))
        model.add(Dropout(drop))

        model.add(Flatten())
        layer0 = Dense(512, activation="relu",kernel_initializer="he_normal",  
                                        kernel_regularizer=keras.regularizers.l2(l2)) # l2(0.01)
        layer1 = Dense(128, activation="relu",kernel_initializer="he_normal",                                                                                        
                                        kernel_regularizer=keras.regularizers.l2(l2))
        layer_output = Dense(len(class_names), activation="softmax",kernel_initializer="glorot_uniform")

        model.add(layer0)
        model.add(Dropout(drop))
        model.add(layer1)
        model.add(Dropout(drop))
        model.add(layer_output)

        # The model’s summary() method displays all the model’s layers
        #model.summary()
        
        return model

    def architecture_VGG16_224(class_names, drop, l2):
        # Creating the model using the Sequential API
        print("[INFO] VGG16 network architecture...")
        model = Sequential()
        #input_shape = train_images[0].shape
        model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Dropout(drop))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Dropout(drop))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Dropout(drop))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Dropout(drop))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Dropout(drop))
        
        model.add(Flatten())
        #model.add(Dense(units=4096,activation="relu"))
        #model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(4096, kernel_regularizer=regularizers.l2(l2), activation="relu")) # l2(0.001)
        model.add(Dropout(drop))
        model.add(Dense(4096, kernel_regularizer=regularizers.l2(l2), activation="relu"))
        model.add(Dropout(drop))
        #model.add(Dense(4096, activation="relu"))
        #model.add(Dropout(0.5))
        #model.add(Dense(4096, activation="relu"))
        #model.add(Dropout(0.5))
        model.add(Dense(len(class_names), activation="softmax"))
        #model.name = 'Dropout layers model'

        # The model’s summary() method displays all the model’s layers
        #model.summary()
        
        return model

    def architecture_VGG16_128(class_names, drop, l2, resolution):
        model = Sequential()
        #input_shape = train_images[0].shape
        #model.add()
        model.add(Conv2D(64, kernel_size=(4,4), activation='relu', padding='same', input_shape=(resolution, resolution, 3)))
        model.add(Conv2D(64, kernel_size=(4,4), activation='relu', padding='same'))
        model.add(Conv2D(64, kernel_size=(4,4), activation='relu', padding='same'))
        #model.add(Conv2D(64, kernel_size=(4,4), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2, 2),strides=(2,2)))
        #model.add(Dropout(0.2))
        
        #model.add(Conv2D(128, kernel_size=(4,4), activation='relu', padding='same'))
        model.add(Conv2D(128, kernel_size=(4,4), activation='relu', padding='same'))
        model.add(Conv2D(128, kernel_size=(4,4), activation='relu', padding='same'))
        model.add(Conv2D(128, kernel_size=(4,4), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2, 2),strides=(2,2)))
        #model.add(Dropout(0.3))
        
        model.add(Conv2D(256, kernel_size=(4,4), activation='relu', padding='same'))
        model.add(Conv2D(256, kernel_size=(4,4), activation='relu', padding='same'))
        model.add(Conv2D(256, kernel_size=(4,4), activation='relu', padding='same'))
        model.add(Conv2D(256, kernel_size=(4,4), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2, 2),strides=(2,2)))
        #model.add(Dropout(0.4))
        
        model.add(Conv2D(filters=512, kernel_size=(4,4), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(4,4), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(4,4), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(4,4), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(2048, activation='relu', kernel_initializer='lecun_uniform', kernel_regularizer=keras.regularizers.l2(l2)))
        model.add(Dropout(0.2))
        model.add(Dense(2048, activation='relu', kernel_initializer='lecun_uniform', kernel_regularizer=keras.regularizers.l2(l2)))
        model.add(Dropout(0.3))
        model.add(Dense(len(class_names), activation='softmax'))
        
        #model.summary()
        
        return model