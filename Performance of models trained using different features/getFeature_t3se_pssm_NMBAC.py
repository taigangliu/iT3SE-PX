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


def get_fasta_list(file):
    f = open(file)
    fastaList = []
    lines = f.readlines()
    L = len(lines)
    for i in range(0, L):
        if i % 2 == 1:
            lines[i] = lines[i].strip()
            fastaList.append(lines[i])
    return fastaList

def getLenList(fastaList):
    lenList=[]
    #i = 0
    for item in fastaList:
        lenList.append(len(item))
    return lenList

def divideMartix(file, len):
    pssm1 = []
    f = open(file)
    martixlines = f.readlines()
    for i in range(3, len + 3):
        pssm = martixlines[i].split()
        pssm1.append(pssm[1:22])
    pssm1 = pd.DataFrame(pssm1)
    seq = pssm1[0]
    matrix = pssm1.iloc[:, 1:22].values
    matrix = np.array(matrix)
    matrix = matrix.astype(int)
    return seq, matrix

def autoNorm(matrix):
    matrix=1.0/(1+np.exp(0-matrix))
    return matrix

def getMatrix(dirname,lenList):
    pssmList = os.listdir(dirname)
    pssmList.sort(key=lambda x: eval(x[:]))
    m = len(pssmList)
    amino = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    reMatrix = np.zeros((m, 200))
    for i in range(m):
        (seq, matrix) = divideMartix(dirname+"/"+pssmList[i],lenList[i])
        matrix_int = np.zeros(matrix.shape)
        matrix_int = matrix.astype(int)
        matrix = autoNorm(matrix_int)
        ls3 = []
        for lag in range(1, 11):
            for j in range(20):
                sum = 0.0
                for z in range(lenList[i] - lag):
                    sum += (matrix[z, j] * matrix[z + lag, j])
                ls3.append(sum / (lenList[i] - lag))
        ls3 = np.array(ls3)
        reMatrix[i] = ls3
    return reMatrix

def main():
    f1 = get_fasta_list("/home/ccding/T3SE_train1491final/T3SE_train1491/dataset/positive.txt")
    lenList1 = getLenList(f1)
    x1 = getMatrix('/home/ccding/T3SE_train1491final/T3SE_train1491/result/positive/pssm_profile_uniref50', lenList1)
    x1 = np.insert(x1, 0, values=[1 for _ in range(x1.shape[0])], axis=1)
    f2 = get_fasta_list("/home/ccding/T3SE_train1491final/T3SE_train1491/dataset/negative.txt")
    m1 = len(f2)
    lenList2 = getLenList(f2)
    x2 = getMatrix('/home/ccding/T3SE_train1491final/T3SE_train1491/result/negative/pssm_profile_uniref50', lenList2)
    x2 = np.insert(x2, 0, values=[-1 for _ in range(x2.shape[0])], axis=1)
    x = np.row_stack((x1, x2))
    x = pd.DataFrame(x)
    train_x = x.drop([0], axis=1).values
    pro_y = x[0].values

    train_x = pd.DataFrame(train_x)
    train_x.to_csv('train_NMBAC2001.csv')

    f1 = get_fasta_list("/home/ccding/T3SE_test216final/T3SE_test216/dataset/positive.txt")
    lenList1 = getLenList(f1)
    x1 = getMatrix('/home/ccding/T3SE_test216final/T3SE_test216/result/positive/pssm_profile_uniref50', lenList1)
    x1 = np.insert(x1, 0, values=[1 for _ in range(x1.shape[0])], axis=1)
    f2 = get_fasta_list("/home/ccding/T3SE_test216final/T3SE_test216/dataset/negative.txt")
    m1 = len(f2)
    lenList2 = getLenList(f2)
    x2 = getMatrix('/home/ccding/T3SE_test216final/T3SE_test216/result/negative/pssm_profile_uniref50', lenList2)
    x2 = np.insert(x2, 0, values=[-1 for _ in range(x2.shape[0])], axis=1)
    x = np.row_stack((x1, x2))
    x = pd.DataFrame(x)
    test_x = x.drop([0], axis=1).values
    pro_py = x[0].values
    test_x = pd.DataFrame(test_x)
    test_x.to_csv('test_NMBAC2001.csv')


main()

