import sys
import numpy as np
import pandas as pd
import math
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, GridSearchCV, LeaveOneOut, cross_val_score
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

def Norm(feature):
    minVals = feature.min(0)
    maxVals = feature.max(0)
    ranges = maxVals-minVals
    m = feature.shape[0]
    feature = feature-np.tile(minVals, (m, 1))
    feature = feature / np.tile(ranges, (m, 1))
    return feature

def main():
    train_x_NMBAC = pd.read_csv('train_NMBAC200.csv')
    test_x_NMBAC = pd.read_csv('test_NMBAC200.csv')
    train_x_NMBAC = np.array(train_x_NMBAC)
    test_x_NMBAC = np.array(test_x_NMBAC)
    train_x_NMBAC = np.delete(train_x_NMBAC, 0, axis=1)
    test_x_NMBAC = np.delete(test_x_NMBAC, 0, axis=1)

    train_x = train_x_NMBAC
    test_x = test_x_NMBAC
    print(train_x.shape)
    print(test_x.shape)

    pro_y = pd.read_csv('train_lable.csv')
    pro_py = pd.read_csv('test_lable.csv')
    pro_y = np.array(pro_y)
    pro_py = np.array(pro_py)
    pro_y = np.delete(pro_y, 0, axis=1)
    pro_py = np.delete(pro_py, 0, axis=1)
    pro_y = pd.DataFrame(pro_y)
    pro_py = pd.DataFrame(pro_py)
    pro_y = pro_y.values.ravel()
    pro_py = pro_py.values.ravel()

    x_all = np.vstack((train_x, test_x))
    x_all = Norm(x_all)
    pro_x = x_all[0:1491, :]
    pro_px = x_all[1491:, :]

    CC = []
    gammas = []
    for i in range(-5, 15, 2):
        CC.append(2 ** i)
    for i in range(3, -15, -2):
        gammas.append(2 ** i)
    param_grid = {"C": CC, "gamma": gammas}
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    gs = GridSearchCV(SVC(probability=True), param_grid, cv=kf)  # 网格搜索
    gs.fit(pro_x, pro_y)
    print(gs.best_estimator_)
    ''''''
    print(gs.best_score_)
    ''''''
    clf = gs.best_estimator_

    acc = []
    sn = []
    sp = []
    f1 = []
    mcc = []

    for t in range(100):
        print('第%d次五折正在进行......' % t)
        cv = KFold(n_splits=5, shuffle=True)
        probass_y = []
        NBtest_index = []
        pred_y = []
        pro_y1 = []
        for train, test in cv.split(pro_x):  # train  test  是下标
            x_train, x_test = pro_x[train], pro_x[test]
            y_train, y_test = pro_y[train], pro_y[test]
            NBtest_index.extend(test)
            probas_ = clf.fit(x_train, y_train).predict_proba(x_test)
            y_train_pred = clf.predict(x_test)
            y_train_probas = clf.predict_proba(x_test)
            probass_y.extend(y_train_probas[:, 1])
            pred_y.extend(y_train_pred)
            pro_y1.extend(y_test)
        cm = confusion_matrix(pro_y1, pred_y)
        tn, fp, fn, tp = cm.ravel()
        ACC = (tp + tn) / (tp + tn + fp + fn)
        SN = tp / (tp + fn)
        SP = tn / (tn + fp)
        PR = tp / (tp + fp)
        MCC = (tp * tn - fp * fn) / math.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))
        F1 = (2 * SN * PR) / (SN + PR)
        # print(MCC)
        acc.append(ACC)
        sn.append(SN)
        sp.append(SP)
        f1.append(F1)
        mcc.append(MCC)
    print(len(acc))
    print('meanACC:', np.mean(acc))
    print('meanSN:', np.mean(sn))
    print('meanSP:', np.mean(sp))
    print('meanF1:', np.mean(f1))
    print('meanMCC:', np.mean(mcc))

    print('stdACC:', np.std(acc))
    print('stdSN:', np.std(sn))
    print('stdSP:', np.std(sp))
    print('stdF1:', np.std(f1))
    print('stdMCC:', np.std(mcc))
main()


#