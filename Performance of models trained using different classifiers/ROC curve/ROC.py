import csv
import numpy as np
import pandas as pd
import math
import os
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, GridSearchCV, LeaveOneOut, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

def getMeanProba(indexFile, ProbaFile):
    GBM_index = pd.read_csv(indexFile, header=None)
    GBM_proba = pd.read_csv(ProbaFile, header=None)
    GBM_index = np.array(GBM_index)
    GBM_proba = np.array(GBM_proba)
    GBM_AUC = []
    r = GBM_index.shape[0]
    c = GBM_index.shape[1]
    mean_proba = []
    GBM_mean_proba1 = []

    for i in range(c):
        sum = 0
        for j in range(r):
            ls = []
            a = GBM_index[j]
            a = np.array(a)
            s = np.argwhere(a == i)
            sum += GBM_proba[j, s]
        mean_proba.extend(sum / 100)
    mean_proba = pd.DataFrame(mean_proba)
    mean_proba = np.array(mean_proba)
    for i in range(mean_proba.shape[0]):
        for j in range(mean_proba.shape[1]):
            GBM_mean_proba1.append(mean_proba[i][j])
    return GBM_mean_proba1

def main():
    train_y = pd.read_csv('train_lable.csv')
    train_y = np.array(train_y)
    lable_ls = train_y[:, 1]
    lable_ls = np.array(lable_ls)
    lable_ls = lable_ls.tolist()

    NB_MeanProba = getMeanProba('NBtest110D_index.csv', 'NBy110D_proba.csv')
    NB_fpr, NB_tpr, NB_thresholds = roc_curve(lable_ls, NB_MeanProba)
    NB_roc_auc = metrics.auc(NB_fpr, NB_tpr)
    print(NB_roc_auc)

    RF_MeanProba = getMeanProba('RFtest110D_index.csv', 'RFy110D_proba.csv')
    RF_fpr, RF_tpr, RF_thresholds = roc_curve(lable_ls, RF_MeanProba)
    RF_roc_auc = metrics.auc(RF_fpr, RF_tpr)
    print(RF_roc_auc)

    SVM_MeanProba = getMeanProba('SVMtest110D_index.csv', 'SVMy110D_proba.csv')
    SVM_fpr, SVM_tpr, SVM_thresholds = roc_curve(lable_ls, SVM_MeanProba)
    SVM_roc_auc = metrics.auc(SVM_fpr, SVM_tpr)
    print(SVM_roc_auc)

    plt.figure()
    plt.plot(NB_fpr, NB_tpr, color='darkslateblue',
             label='NB (AUC = %0.3f)' %  NB_roc_auc)

    plt.plot(RF_fpr, RF_tpr, color='darkturquoise',
             label='RF (AUC = %0.3f)' % RF_roc_auc)

    plt.plot(SVM_fpr, SVM_tpr, color='darkred',
             label='SVM (AUC = %0.3f)' % SVM_roc_auc)

    plt.plot([0, 1], [0, 1], color='darkcyan', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate ')
    plt.ylabel('True Positive Rate ')
    plt.grid(True)
    plt.legend(loc="best")
    plt.show()

main()





