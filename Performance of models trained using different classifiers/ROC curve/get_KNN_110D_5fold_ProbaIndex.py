import csv
import sys
import numpy as np
import pandas as pd
import math
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold, GridSearchCV, LeaveOneOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier


def main():

    train_x_NMBAC = pd.read_csv('110D_train.csv')
    test_x_NMBAC = pd.read_csv('110D_test.csv')
    train_x_NMBAC = np.array(train_x_NMBAC)
    test_x_NMBAC = np.array(test_x_NMBAC)
    train_x_NMBAC = np.delete(train_x_NMBAC, 0, axis=1)
    test_x_NMBAC = np.delete(test_x_NMBAC, 0, axis=1)

    train_x = train_x_NMBAC
    pro_px = test_x_NMBAC
    print(train_x.shape)
    print(pro_px.shape)

    pro_y = pd.read_csv('train_lable.csv')
    pro_py = pd.read_csv('test_lable.csv')
    pro_y = np.array(pro_y)
    pro_py = np.array(pro_py)
    pro_y = np.delete(pro_y, 0, axis=1)
    pro_py = np.delete(pro_py, 0, axis=1)
    pro_y = pd.DataFrame(pro_y)
    pro_py = pd.DataFrame(pro_py)
    train_y = pro_y.values.ravel()
    pro_py = pro_py.values.ravel()

    param_grid = {'n_neighbors': [1, 3, 7], 'p': [1, 2]}
    kf = KFold(n_splits=5, shuffle=True)
    gs = GridSearchCV(KNeighborsClassifier(), param_grid, 'accuracy', cv=kf)
    gs.fit(train_x, train_y)
    print(gs.best_estimator_)
    print(gs.best_score_)
    clf = gs.best_estimator_

    for t in range(100):
        print('第%d次五折正在进行......' % t)
        cv = KFold(n_splits=5, shuffle=True)
        acc = []
        sn = []
        sp = []
        mcc = []
        f1 = []
        probass_y = []
        GBMtest_index = []
        for train, test in cv.split(train_x, train_y):
            GBMtest_index.extend(test)
            probas_ = clf.fit(train_x[train], train_y[train]).predict_proba(train_x[test])
            y_train_pred = clf.predict(train_x[test])
            y_train_probas = clf.predict_proba(train_x[test])
            probass_y.extend(y_train_probas[:, 1])
            auc = roc_auc_score(train_y[test], y_train_probas[:, 1])


        csvFile = open("KNNy110D_proba.csv", 'a', newline='')
        writer = csv.writer(csvFile, dialect='excel')
        writer.writerow(probass_y)
        csvFile.close()

        csvFile = open("KNNtest110D_index.csv", 'a', newline='')
        writer = csv.writer(csvFile, dialect='excel')
        writer.writerow(GBMtest_index)
        csvFile.close()


main()


