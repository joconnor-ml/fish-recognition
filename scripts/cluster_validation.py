import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, GroupKFold
from sklearn.cluster import DBSCAN, KMeans
from sklearn.utils import shuffle
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA

architecture = "vgg16"
alb, bet, dol, lag, nof, oth, sha, yft = 1719, 200, 117, 67, 465, 299, 176, 734
df = pd.read_csv("../data/train.bottleneck.{}.csv".format(architecture), index_col=0)

y = np.array([0] * alb + [1] * bet + [2] * dol + [3] * lag + [4] * nof + [5] * oth + [6] * sha + [7] * yft)
X = df.copy()
X, y = shuffle(X, y)

X_pca = PCA(3).fit_transform(X)
kmeans = KMeans(10)
clusters = kmeans.fit(X_pca).predict(X_pca)
# print(kmeans.score(X_pca))
# print(clusters)

dftest = pd.read_csv("../data/test.bottleneck.{}.csv".format(architecture), index_col=0)
dftest.head()

p = []
pt = []
lls = []
cv = GroupKFold(2)
print("Group KFold")
for i, (train, test) in enumerate(cv.split(X, y, clusters)):
    model = LogisticRegression(C=1e-4).fit(X.iloc[train], y[train])
    ll = log_loss(y[test], model.predict_proba(X.iloc[test]), labels=[0, 1, 2, 3, 4, 5, 6, 7])
    print(ll)
    p.append((y[test], model.predict_proba(X.iloc[test])))
    pt.append(model.predict_proba(dftest))
    lls.append(ll)

cv = StratifiedKFold(2)
print("Stratified KFold")
for i, (train, test) in enumerate(cv.split(X, y)):
    model = LogisticRegression(C=1e-4).fit(X.iloc[train], y[train])
    ll = log_loss(y[test], model.predict_proba(X.iloc[test]), labels=[0, 1, 2, 3, 4, 5, 6, 7])
    print(ll)
    p.append((y[test], model.predict_proba(X.iloc[test])))
    pt.append(model.predict_proba(dftest))
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
