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
    train_x_NMBAC = pd.read_csv('110D_train.csv')
    test_x_NMBAC = pd.read_csv('110D_test.csv')
    train_x_NMBAC = np.array(train_x_NMBAC)
    test_x_NMBAC = np.array(test_x_NMBAC)
    train_x_NMBAC = np.delete(train_x_NMBAC, 0, axis=1)
    test_x_NMBAC = np.delete(test_x_NMBAC, 0, axis=1)

    pro_x = train_x_NMBAC
    pro_px = test_x_NMBAC
    print(pro_x.shape)
    print(pro_px.shape)

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

    clf = SVC(C=2, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.5, kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
    verbose=False)

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


