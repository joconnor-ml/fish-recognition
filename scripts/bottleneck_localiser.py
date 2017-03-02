import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneLabelOut, cross_val_score, cross_val_predict, LabelKFold
from sklearn.cluster import DBSCAN, KMeans
from sklearn.utils import shuffle
from sklearn.metrics import log_loss

from .data_utils import get_locations_train

architecture = "alexnet"
C = 0.001
alb, bet, dol, lag, nof, oth, sha, yft = 1719, 200, 117, 67, 465, 299, 176, 734

df = pd.read_csv("train.bottleneck.{}.csv".format(architecture), index_col=0)

y = get_locations_train()

X = df.copy()
X, y, cl = shuffle(X, y, cl)

dftest = pd.read_csv("test.bottleneck.{}.csv".format(architecture), index_col=0)
dftest.head()

p = []
pt = []
cv = StratifiedKFold(8)
for i, (train, test) in enumerate(cv.split(X, y, cl)):
    print(np.unique(y[train]))
    print(X.iloc[train].shape, y[train].shape)
    model = Lasso(C=C).fit(X.iloc[train], y[train])
    print(log_loss(y[test], model.predict_proba(X.iloc[test]), labels=[0, 1, 2, 3, 4, 5, 6, 7]))
    p.append((y[test], model.predict_proba(X.iloc[test])))
    pt.append(model.predict_proba(dftest))

model = Lasso(C=C).fit(X, y)
p = model.predict_proba(dftest.values)
sub = pd.read_csv("sample_submission_stg1.csv", index_col=0)
samp = sub.copy()
print(samp.mean(axis=0))

sub.loc[dftest.index, :] = p
print(sub.mean(axis=0))

sub.to_csv("sub.lr{:.5f}.bottleneck_{}.csv.gz".format(C, architecture), compression="gzip")

print(sub.head())
sub.loc[dftest.index, :] = pm.mean(axis=0)
sub.to_csv("sub.lr{:.5f}.8fold.bottleneck_{}.csv.gz".format(C, architecture), compression="gzip")
