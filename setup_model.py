## setup_model.py -- gtsrb data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import os
import pathlib
import pandas as pd
import numpy as np
import random
import cv2
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization

def extract_data(data_dir, npy_path):
  image_data = []
  image_labels = []

  for i in range(43):
      path = os.path.join(data_dir,'Train',str(i))
      images = os.listdir(path)

      for img in images:
          try:
              image = cv2.imread(path + '/' + img)
              image_fromarray = Image.fromarray(image, 'RGB')
              resize_image = image_fromarray.resize((30, 30))
              image_data.append(np.array(resize_image))
              image_labels.append(i)
          except:
              print("Error in " + img)

  # Changing the list to numpy array
  image_data = np.array(image_data)
  image_labels = np.array(image_labels)

  # Splitting the dataset into training and validation set
  X_train, X_val, y_train, y_val = train_test_split(image_data, image_labels, test_size=0.3, random_state=42)

  X_train = X_train.astype('float32')/255
  X_val = X_val.astype('float32')/255

  y_train = to_categorical(y_train, 43)
  y_val = to_categorical(y_val, 43)

  X_test = np.load(npy_path)
  X_test = X_test.astype('float32')/255
  y_test = pd.read_csv(data_dir + '/Test.csv')
  y_test = y_test["ClassId"].values.tolist()
  y_test = np.array(y_test)
  y_test = to_categorical(y_test, 43)

  return X_train, X_val, y_train, y_val, X_test, y_test

class GTSRB:
    def __init__(self):
        X_train, X_val, y_train, y_val, X_test, y_test = extract_data('/Users/damiafatihah/Desktop/Data Science Project/TrafficSignRecognition/Dataset', '/Users/damiafatihah/Desktop/Data Science Project/TrafficSignRecognition/Dataset/Saved Test Data/array_2.npy')

        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

class Model:
    def __init__(self, restore, session=None):
        self.num_channels = 3
        self.image_size = 30
        self.num_labels = 43

        model = Sequential()

        model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(30,30,3)))
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

        self.model = model

        if restore is not None:
            self.model.load_weights(restore)

    def predict(self, data):
        return self.model(data)