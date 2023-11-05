## test_attack.py -- sample code to test attack procedure
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
import time
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from l2_attack import CarliniL2
from setup_model import GTSRB, Model

def show(img):
    """
    Display 32x32 traffic sign image.
    """
    if img.shape != (30, 30, 3):  # Check if image is 30x30x3
        print("Invalid image dimensions!")
        return

    plt.imshow(img)
    plt.show()


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.y_test.shape[1])

            for j in seq:
                if (j == np.argmax(data.y_test[start+i])) and (inception == False):
                    continue
                inputs.append(data.X_test[start+i])
                targets.append(np.eye(data.y_test.shape[1])[j])
        else:
            inputs.append(data.X_test[start+i])
            targets.append(data.y_test[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets

if __name__ == "__main__":
  with tf.Session() as sess:
    data, model =  GTSRB(), Model('/Users/damiafatihah/Desktop/Data Science Project/TrafficSignRecognition/Model/model_without_softmax.h5')
    attack = CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=0, boxmin=0, boxmax=1)
        #attack = CarliniL0(sess, model, max_iterations=1000, initial_const=10,
        #                   largest_const=15)

    inputs, targets = generate_data(data, samples=10, targeted=True,
                                        start=0, inception=False)
    timestart = time.time()
    adv = attack.attack(inputs, targets)
    timeend = time.time()

    print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

    for i in range(len(adv)):
        print("Valid:")
        show(inputs[i])
        print("Adversarial:")
        show(adv[i])

        print("Classification:", model.model.predict(adv[i:i+1]))

        print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)