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

architecture = "vgg16"
alb, bet, dol, lag, nof, oth, sha, yft = 1719, 200, 117, 67, 465, 299, 176, 734
df = pd.read_csv("../data/train.bottleneck.{}.csv".format(architecture), index_col=0)

y = np.array([0] * alb + [1] * bet + [2] * dol + [3] * lag + [4] * nof + [5] * oth + [6] * sha + [7] * yft)
X = df.copy().values
X, y = shuffle(X, y)
y = to_categorical(y)
print(y[0])

X_pca = PCA(3).fit_transform(X)
kmeans = KMeans(10)
clusters = kmeans.fit(X_pca).predict(X_pca)
# print(kmeans.score(X_pca))
# print(clusters)

dftest = pd.read_csv("../data/test.bottleneck.{}.csv".format(architecture), index_col=0)
dftest.head()

def get_model():
    inp = Input(X.shape[1:])
    x = Dense(512, activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    #x_bb = Dense(4, name='bb')(x)
    x_class = Dense(8, activation='softmax', name='class')(x)
    model = Model(input=inp, output=x_class)
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


p = []
pt = []
lls = []
cv = GroupKFold(2)
print("Group KFold")
for i, (train, test) in enumerate(cv.split(X, groups=clusters)):
    print(i)
    model = get_model()
    model.fit(X[train, :], y[train, :], nb_epoch=10, validation_data=(X[test, :], y[test, :]))#, batch_size=32)
    ll = log_loss(y[test], model.predict(X[test]))
    print(ll)
    p.append((y[test], model.predict(X[test])))
    pt.append(model.predict(dftest.values))
    lls.append(ll)

cv = StratifiedKFold(2)
print("Stratified KFold")
for i, (train, test) in enumerate(cv.split(X, y[:,0])):
    model = get_model()
    model.fit(X[train, :], y[train, :], nb_epoch=10, validation_data=(X[test, :], y[test, :]))#, batch_size=32)
    ll = log_loss(y[test], model.predict(X[test]))
    print(ll)
    p.append((y[test], model.predict(X[test])))
    pt.append(model.predict(dftest.values))
    lls.append(ll)

print("Average")
print(sum(lls)/len(lls))

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

