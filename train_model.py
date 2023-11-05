## train_models.py -- train the neural network models for attacking
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import os
import pathlib
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image
from setup_model import GTSRB
from matplotlib.image import imread
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization

def train(data, file_name, num_epochs=30, batch_size=32, IMG_HEIGHT=30, IMG_WIDTH=30, channels=3, init=None):
    """
    Standard neural network training procedure.
    """
    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,channels)))
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=-1))

    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=-1))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))

    model.add(Dense(43))

    if init != None:
        model.load_weights(init)

    lr = 0.001
    epochs = 30

    opt = Adam(lr=lr, decay=lr / (epochs * 0.5))
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, name='categorical_crossentropy'), optimizer=opt, metrics=['accuracy'])

    aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

    model.fit(aug.flow(data.X_train, data.y_train, batch_size=batch_size), epochs=num_epochs, validation_data=(data.X_val, data.y_val))

    if file_name != None:
        model.save(file_name)

    return model

# def train_distillation(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1):
#     """
#     Train a network using defensive distillation.

#     Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks
#     Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, Ananthram Swami
#     IEEE S&P, 2016.
#     """
#     if not os.path.exists(file_name+"_init"):
#         # Train for one epoch to get a good starting point.
#         train(data, file_name+"_init", params, 1, batch_size)

#     # now train the teacher at the given temperature
#     teacher = train(data, file_name+"_teacher", params, num_epochs, batch_size, train_temp,
#                     init=file_name+"_init")

#     # evaluate the labels at temperature t
#     predicted = teacher.predict(data.train_data)
#     with tf.Session() as sess:
#         y = sess.run(tf.nn.softmax(predicted/train_temp))
#         print(y)
#         data.train_labels = y

#     # train the student model at temperature t
#     student = train(data, file_name, params, num_epochs, batch_size, train_temp,
#                     init=file_name+"_init")

#     # and finally we predict at temperature 1
#     predicted = student.predict(data.train_data)

#     print(predicted)


# train(CIFAR(), "models/cifar", num_epochs=30)
train(GTSRB(), "/Users/damiafatihah/Desktop/Data Science Project/TrafficSignRecognition/Model/model_without_softmax.h5")

# train_distillation(MNIST(), "models/mnist-distilled-100", [32, 32, 64, 64, 200, 200],
#                    num_epochs=50, train_temp=100)
# train_distillation(CIFAR(), "models/cifar-distilled-100", [64, 64, 128, 128, 256, 256],
#                    num_epochs=50, train_temp=100)