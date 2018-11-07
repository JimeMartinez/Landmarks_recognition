# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 23:14:42 2018

@author: jimena
"""

from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import optimizers
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
session_conf.gpu_options.allow_growth = True
session_conf.gpu_options.allocator_type ='BFC'
K.set_session(sess)

num_classes = 17
image_size = 256

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(image_size,image_size,3), kernel_initializer='glorot_uniform'))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='glorot_uniform'))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='glorot_uniform'))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer='glorot_uniform'))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer='glorot_uniform'))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer=optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary

data_generator_train = ImageDataGenerator(rescale=1./127.5,
                                          rotation_range=20,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          horizontal_flip=True)
data_generator_test = ImageDataGenerator(rescale=1./127.5)

train_generator = data_generator_train.flow_from_directory(
        '../TFM/Dataset_Resize_Split_Train',
        target_size=(image_size, image_size),
        batch_size=32,
        class_mode='categorical')

validation_generator = data_generator_test.flow_from_directory(
        '../TFM/Dataset_Resize_Split_Dev',
        target_size=(image_size, image_size),
        batch_size=32,
        class_mode='categorical')

tbCallBack =  TensorBoard(log_dir='/Tensorboard/My_Model_8', histogram_freq=0, write_graph=True, write_images=True)
early_stopping = EarlyStopping(monitor='val_acc', patience=3)

model.fit_generator(
        train_generator,
        epochs=20,
        callbacks=[tbCallBack, early_stopping, ModelCheckpoint('model_8.h5', save_best_only=True)],
        validation_data=validation_generator)

K.clear_session()