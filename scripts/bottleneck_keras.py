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
kmeans = KMeans(3)
clusters = kmeans.fit(X_pca).predict(X_pca)
df["cluster3"] = clusters
kmeans = KMeans(10)
clusters = kmeans.fit(X_pca).predict(X_pca)
df["cluster10"] = clusters
kmeans = KMeans(100)
clusters = kmeans.fit(X_pca).predict(X_pca)
df["cluster100"] = clusters
y_clust = to_categorical(clusters)
print(y[0], y_box[0], y_clust[0])
print(kmeans.score(X_pca))
# print(clusters)

df.to_csv("../data/train.bottleneck.{}.targets.csv.gz".format(architecture), compression="gzip")

dftest = pd.read_csv("../data/test.bottleneck.{}.csv".format(architecture), index_col=0)

dftest.head()

def get_model():
    inp = Input(X.shape[1:])
    x = Dropout(0.15)(inp)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x_box = Dense(4, name='bb')(x)
    x_class = Dense(8, activation='softmax', name='class')(x)
    model = Model(input=inp, output=[x_class, x_box])
    model.compile(Adam(lr=0.001), loss=['categorical_crossentropy', 'mse'],
                  metrics=['accuracy'], loss_weights=[1., 0.001])
    return model

def get_clust_model():
    inp = Input(X.shape[1:])
    x = Dropout(0.15)(inp)
    x = Dense(512, activation='relu')(x)
    x_clust = Dense(10, activation='softmax', name='clust')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x_box = Dense(4, name='bb')(x)
    x_class = Dense(8, activation='softmax', name='class')(x)
    model = Model(input=inp, output=[x_class, x_box, x_clust])
    model.compile(Adam(lr=0.0001), loss=['categorical_crossentropy', 'mse', 'categorical_crossentropy'],
                  metrics=['accuracy'], loss_weights=[1., 0.001, 0.1])
    return model


p = []
pt = []
lls = []
cv = GroupKFold(3)

print("Group KFold")
for i, (train, test) in enumerate(cv.split(X, groups=clusters)):
    print(i)
    model = get_model()
    model.fit(X[train, :], [y[train, :], y_box[train, :]], nb_epoch=20, validation_data=(X[test, :], [y[test, :], y_box[test, :]]), batch_size=128, verbose=2)
    test_preds = model.predict(X[test])[0].clip(0.1, 0.8)
    print(test_preds)
    ll = log_loss(y[test], test_preds)
    print(ll)
    #p.append((y[test], model.predict(X[test])[:, -1]))
    pt.append(model.predict(dftest.values)[0])
    lls.append(ll)

#cv = StratifiedKFold(10)
#print("Stratified KFold")
#for i, (train, test) in enumerate(cv.split(X, y[:,0])):
#    print(i)
#    model = get_model()
#    model.fit(X[train, :], [y[train, :], y_box[train, :]], nb_epoch=20, validation_data=(X[test, :], [y[test, :], y_box[test, :]]), batch_size=128, verbose=2)
#    test_preds = model.predict(X[test])[0].clip(0.025, 0.875)
#    print(test_preds)
#    ll = log_loss(y[test], test_preds)
#    print(ll)
#    #p.append((y[test], model.predict(X[test])[:, -1]))
#    pt.append(model.predict(dftest.values)[0])
#    lls.append(ll)


print("Average")
print(sum(lls)/len(lls))

folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
test = pd.DataFrame(sum(pt) / len(pt), index=dftest.index, columns=folders)
test.index.name = "image"
test.clip(0.05, 0.85).to_csv("sub.csv.gz", compression="gzip")

#model = Lasso(C=C).fit(X, y)
#p = model.predict_proba(dftest.values)
#sub = pd.read_csv("sample_submission_stg1.csv", index_col=0)
#samp = sub.copy()
#print(samp.mean(axis=0))

#sub.loc[dftest.index, :] = p
#print(sub.mean(axis=0))

#sub.to_csv("sub.lr{:.5f}.bottleneck_{}.csv.gz".format(C, architecture), compression="gzip")

#print(sub.head())
#sub.loc[dftest.index, :] = pm.mean(axis=0)
#sub.to_csv("sub.lr{:.5f}.8fold.bottleneck_{}.csv.gz".format(C, architecture), compression="gzip")

