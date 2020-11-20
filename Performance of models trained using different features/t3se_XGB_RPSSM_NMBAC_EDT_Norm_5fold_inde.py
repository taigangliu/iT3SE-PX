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

    train_x_RPSSM = pd.read_csv('train_RPSSM1010.csv')
    test_x_RPSSM = pd.read_csv('test_RPSSM1010.csv')
    train_x_RPSSM = np.array(train_x_RPSSM)
    test_x_RPSSM = np.array(test_x_RPSSM)
    train_x_RPSSM = np.delete(train_x_RPSSM, 0, axis=1)
    test_x_RPSSM = np.delete(test_x_RPSSM, 0, axis=1)

    train_x_EDT = pd.read_csv('train_EDT4000.csv')
    test_x_EDT = pd.read_csv('test_EDT4000.csv')
    train_x_EDT  = np.array(train_x_EDT )
    test_x_EDT  = np.array(test_x_EDT )
    train_x_EDT  = np.delete(train_x_EDT , 0, axis=1)
    test_x_EDT  = np.delete(test_x_EDT , 0, axis=1)

    train_x = np.hstack((train_x_NMBAC, train_x_RPSSM))
    train_x = np.hstack((train_x, train_x_EDT))
    test_x = np.hstack((test_x_NMBAC, test_x_RPSSM))
    test_x = np.hstack((test_x, test_x_EDT))
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

    model = XGBClassifier(random_state=eval(sys.argv[1]))
    model.fit(pro_x, pro_y)
    import_level = model.feature_importances_
    index = np.argsort(import_level)[::-1]
    rank_matrix = np.zeros((pro_x.shape[0], pro_x.shape[1]))
    rank_matrix_test = np.zeros((pro_px.shape[0], pro_px.shape[1]))

    for i in range(pro_x.shape[1]):
        rank_matrix[:, i] = pro_x[:, index[i]]
        rank_matrix_test[:, i] = pro_px[:, index[i]]
    fold5score = []
    inde = []
    for lag in range(10, 801, 10):
        print('特征个数为%d个时所有的指标情况' % lag)
        CC = []
        gammas = []
        for i in range(-5, 15, 2):
            CC.append(2 ** i)
        for i in range(3, -15, -2):
            gammas.append(2 ** i)
        param_grid = {"C": CC, "gamma": gammas}
        kf = KFold(n_splits=5, shuffle=True, random_state=123)
        gs = GridSearchCV(SVC(probability=True), param_grid, cv=kf)
        gs.fit(rank_matrix[:, 0:lag], pro_y)
        print(gs.best_estimator_)
        ''''''
        print(gs.best_score_)
        ''''''
        clf = gs.best_estimator_
        xtr = rank_matrix[:, 0:lag]
        cv = KFold(n_splits=5, shuffle=True, random_state=123)
        probass_y = []
        test_index = []
        pred_y = []
        pro_5y = []

        for train, test in cv.split(xtr):
            x_train, x_test = xtr[train], xtr[test]
            y_train, y_test = pro_y[train], pro_y[test]
            test_index.extend(test)
            probas_ = clf.fit(x_train, y_train).predict_proba(x_test)
            y_train_pred = clf.predict(x_test)
            y_train_probas = clf.predict_proba(x_test)
            probass_y.extend(y_train_probas[:, 1])
            pred_y.extend(y_train_pred)
            pro_5y.extend(y_test)

        cm = confusion_matrix(pro_5y, pred_y)
        tn, fp, fn, tp = cm.ravel()
        ACC = (tp + tn) / (tp + tn + fp + fn)
        SN = tp / (tp + fn)
        SP = tn / (tn + fp)
        PR = tp / (tp + fp)
        MCC = (tp * tn - fp * fn) / math.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))
        F1 = (2 * SN * PR) / (SN + PR)
        print('5折准确率：', ACC)
        print('SN:', SN)
        print('SP:', SP)
        print('MCC:', MCC)
        print('F1:', F1)
        fold5score.append(ACC)

        clf.fit(xtr, pro_y)
        pro_y_pred = clf.predict(rank_matrix_test[:, 0:lag])
        pro_y_probas = clf.predict_proba(rank_matrix_test[:, 0:lag])
        acc = accuracy_score(pro_py, pro_y_pred)
        inde.append(acc)
        print('独立集结果：', acc)
    print(fold5score)
    print(inde)

main()


