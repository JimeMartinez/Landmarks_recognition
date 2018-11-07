from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import optimizers
import numpy as np
import tensorflow as tf
import random as rn

# Force TensorFlow to use single thread. Multiple threads are a potential source of non-reproducible results.
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K


sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

model = load_model('model_VGG19.h5')

data_generator_test = ImageDataGenerator()

test_generator = data_generator_test.flow_from_directory(
        '../TFM/Dataset_Resize_Split_Test',
        target_size=(512, 512),
        batch_size=24,
        class_mode='categorical')

score = model.evaluate_generator(test_generator)
print(score)


K.clear_session()