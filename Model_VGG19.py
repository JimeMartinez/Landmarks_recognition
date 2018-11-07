# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 18:55:40 2018

@author: jimena
"""

from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary in Python 3.2.3 onwards to have reproducible behavior for certain hash-based operations.

import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers in a well-defined initial state.
np.random.seed(42)

# The below is necessary for starting core Python generated random numbers in a well-defined state.
rn.seed(12345)

# Force TensorFlow to use single thread. Multiple threads are a potential source of non-reproducible results.
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K

# The below tf.set_random_seed() will make random number generation in the TensorFlow backend have a well-defined initial state.
tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

num_classes = 17
image_size = 512


my_new_model = Sequential()
my_new_model.add(VGG19(include_top=False, pooling='avg', weights='imagenet'))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer model. It is already trained
my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

data_generator_train = ImageDataGenerator()
data_generator_test = ImageDataGenerator()

train_generator = data_generator_train.flow_from_directory(
        '../TFM/Dataset_Resize_Split_Train',
        target_size=(image_size, image_size),
        batch_size=24,
        class_mode='categorical')

validation_generator = data_generator_test.flow_from_directory(
        '../TFM/Dataset_Resize_Split_Dev',
        target_size=(image_size, image_size),
        batch_size=24,
        class_mode='categorical')

tbCallBack =  TensorBoard(log_dir='/Tensorboard/Graph_VGG19_Final', histogram_freq=0, write_graph=True, write_images=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

my_new_model.fit_generator(
        train_generator,
        epochs=20,
        callbacks=[tbCallBack, early_stopping, ModelCheckpoint('model_VGG19.h5', monitor='val_acc', save_best_only=True)],
        validation_data=validation_generator)

K.clear_session()