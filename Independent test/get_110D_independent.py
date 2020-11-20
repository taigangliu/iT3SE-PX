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


def main():
    train_x = pd.read_csv('110D_train.csv')
    test_x = pd.read_csv('110D_test.csv')
    train_x = np.array(train_x)
    test_x = np.array(test_x)
    pro_x = np.delete(train_x, 0, axis=1)
    pro_px = np.delete(test_x, 0, axis=1)

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

    S_ACC = []
    S_SN = []
    S_SP = []
    S_F1 = []
    S_MCC = []

    clf.fit(pro_x, pro_y)
    cv = KFold(n_splits=5, shuffle=True)
    for train, test in cv.split(pro_px, pro_py):  # train  test  是下标
        pro_y_pred = clf.predict(pro_px[test])
        pro_y_probas = clf.predict_proba(pro_px[test])
        cm = confusion_matrix(pro_py[test], pro_y_pred)
        tn, fp, fn, tp = cm.ravel()
        ACC = (tp + tn) / (tp + tn + fp + fn)
        SN = tp / (tp + fn)
        SP = tn / (tn + fp)
        PR = tp / (tp + fp)
        MCC = (tp * tn - fp * fn) / math.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))
        F1 = (2 * SN * PR) / (SN + PR)
        S_ACC.append(ACC)
        S_SN.append(SN)
        S_SP.append(SP)
        S_F1.append(F1)
        S_MCC.append(MCC)
    print(np.mean(S_ACC))
    print(np.mean(S_SN))
    print(np.mean(S_SP))
    print(np.mean(S_F1))
    print(np.mean(S_MCC))

    print(np.std(S_ACC))
    print(np.std(S_SN))
    print(np.std(S_SP))
    print(np.std(S_F1))
    print(np.std(S_MCC))

main()

