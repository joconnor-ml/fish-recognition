"""See how far we can get just by training on the average intensity in each colour
channel and the image shape -- another proxy for boat ID"""
import cv2
import glob
import os
import pandas as pd
import numpy as np
import time
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer

def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train files')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('..', 'input', 'train', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_features(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id

def load_test():
    path = os.path.join('..', 'input', 'test_stg1', '*.jpg')
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_features(fl)
        X_test.append(img)
        X_test_id.append(flbase)

    return X_test, X_test_id

def get_features(path):
    img = cv2.imread(path)
    return np.concatenate([img.shape, img.mean(axis=0).mean(axis=0)])

def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)

nfolds = 3
X, y, Xid = load_train()

for c in [100, 10, 5, 1, 0.5, 0.1, 0.01]:
    print(c)
    model = LogisticRegression(C=c)
    preds = cross_val_predict(model, X, y, cv=StratifiedKFold(nfolds), method="predict_proba")
    loss = log_loss(y, preds)
    print(loss)

print("XGB")
model = XGBClassifier()
preds = cross_val_predict(model, np.array(X), y, cv=StratifiedKFold(nfolds), method="predict_proba")
loss = log_loss(y, preds)
print(loss)

#info_string = "logistic_regression_loss_{:.4f}".format(loss)
#model.fit(X, y)
#X, Xid = load_test()
#preds = model.predict_proba(X)
#create_submission(preds, Xid, info_string)
