import time
from keras.optimizers import SGD
from convnetskeras.convnets import preprocess_image_batch, convnet
from convnetskeras.imagenet_tool import synset_to_dfs_ids

import os
import h5py
import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense


import glob
import cv2
import datetime


def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('..', 'input', 'train', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            im = preprocess_image_batch([fl],
                                        img_size=(256,256),
                                        crop_size=(224,224),
                                        color_mode="bgr")
            out = model.predict(im)
            X_train.append(out.flatten())
            X_train_id.append(flbase)
            y_train.append(index)
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return np.array(X_train), y_train, X_train_id


def load_test():
    path = os.path.join('..', 'input', 'test_stg1', '*.jpg')
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        im = preprocess_image_batch([fl],
                                    img_size=(256,256),
                                    crop_size=(224,224),
                                    color_mode="bgr")
        out = model.predict(im)
        X_test.append(out.flatten())
        X_test_id.append(flbase)
    return np.array(X_test), X_test_id

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model = convnet('vgg_16',weights_path="weights/vgg16_weights.h5")
model.layers.pop()
model.layers.pop()
model.outputs = [model.layers[-1].output]
model.layers[-1].outbound_nodes = []
model.compile(optimizer=sgd, loss='mse')

p, y, Xid = load_train()
print(p)
train = pd.DataFrame(p, index=Xid)

pt, Xtid = load_test()
test = pd.DataFrame(pt, index=Xtid)

train.to_csv("train.bottleneck.vgg16.new.csv")
test.to_csv("test.bottleneck.vgg16.new.csv")
