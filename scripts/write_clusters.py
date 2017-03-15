import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, GroupKFold
from sklearn.cluster import DBSCAN, KMeans
from sklearn.utils import shuffle
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils.layer_utils import layer_from_config
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer

import utils

np.random.seed(10)

architecture = "vgg16"
alb, bet, dol, lag, nof, oth, sha, yft = 1719, 200, 117, 67, 465, 299, 176, 734
df = pd.read_csv("../data/train.bottleneck.{}.csv".format(architecture), index_col=0)
df["target"] = np.array([0] * alb + [1] * bet + [2] * dol + [3] * lag + [4] * nof + [5] * oth + [6] * sha + [7] * yft)
boxes = utils.read_boxes()
df = df.join(boxes).fillna(0)
print(df.head())

y=df["target"].values
y_box = df[["height", "width", "x", "y"]].values
X = df.drop(["height", "width", "x", "y", "target"], axis=1).values
X, y, y_box = shuffle(X, y, y_box)

y = to_categorical(y)


X_pca = PCA(3).fit_transform(X)
kmeans3 = KMeans(3)
clusters = kmeans3.fit(X_pca).predict(X_pca)
df["cluster3"] = clusters
kmeans10 = KMeans(10)
clusters = kmeans10.fit(X_pca).predict(X_pca)
df["cluster10"] = clusters
kmeans100 = KMeans(100)
clusters = kmeans100.fit(X_pca).predict(X_pca)
df["cluster100"] = clusters
y_clust = to_categorical(clusters)
print(y[0], y_box[0], y_clust[0])
print(kmeans3.score(X_pca)/len(X_pca))
print(kmeans10.score(X_pca)/len(X_pca))
print(kmeans100.score(X_pca)/len(X_pca))
# print(clusters)

df[["height", "width", "x", "y", "target", "cluster3", "cluster10", "cluster100"]].to_csv("../data/train.bottleneck.{}.targets.csv.gz".format(architecture), compression="gzip")
del df

dftest = pd.read_csv("../data/test.bottleneck.{}.csv".format(architecture), index_col=0)
Xt = dftest.values
Xt_pca = PCA(3).fit_transform(Xt)
dftest["cluster3"] = kmeans3.predict(Xt_pca)
dftest["cluster10"] = kmeans10.predict(Xt_pca)
dftest["cluster100"] = kmeans100.predict(Xt_pca)
print(kmeans3.score(Xt_pca)/len(Xt_pca))
print(kmeans10.score(Xt_pca)/len(Xt_pca))
print(kmeans100.score(Xt_pca)/len(Xt_pca))
dftest[["cluster3", "cluster10", "cluster100"]].to_csv("../data/test.bottleneck.{}.targets.csv.gz".format(architecture), compression="gzip")
