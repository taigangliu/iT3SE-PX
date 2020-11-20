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
    reMatrix=np.zeros((m,1010))
    for i in range(m):
        (seq, matrix) = divideMartix(dirname+"/"+pssmList[i],lenList[i])
        matrix_int = np.zeros(matrix.shape)
        matrix_int = matrix.astype(int)
        matrix = autoNorm(matrix_int)
        matrix = pd.DataFrame(matrix)
        reMatrix1 = np.zeros((lenList[i], 10))
        reMatrix1 = pd.DataFrame(reMatrix1)
        reMatrix1[0] = (matrix[13] + matrix[17] + matrix[18]) / 3
        reMatrix1[1] = (matrix[13] + matrix[11]) / 2
        reMatrix1[2] = (matrix[9]) + matrix[19] / 2
        reMatrix1[3] = (matrix[0] + matrix[15] + matrix[16]) / 3
        reMatrix1[4] = (matrix[2] + matrix[8]) / 2
        reMatrix1[5] = (matrix[5] + matrix[6] + matrix[3]) / 3
        reMatrix1[6] = (matrix[1] + matrix[11]) / 2
        reMatrix1[7] = matrix[4]
        reMatrix1[8] = matrix[7]
        reMatrix1[9] = matrix[14]
        reMatrix1 = np.array(reMatrix1)
        ls1 = []
        for j in range(10):
            sum1 = 0.0
            sum = 0.0
            for k in range(lenList[i]):
                sum1 = sum1+reMatrix1[k, j]
            avg_j = sum1 / lenList[i]
            for s in range(lenList[i]):
                sum = sum + (reMatrix1[s, j]-avg_j)*(reMatrix1[s, j]-avg_j)
            ls1.append(sum/lenList[i])
        ls2 = []
        for lag in range(1, 11):
            for a in range(10):
                for b in range(10):
                    sum = 0.0
                    for c in range(lenList[i]-lag):
                        sum += (pow(reMatrix1[c, a]-reMatrix1[c+lag, b], 2)) / 2
                    ls2.append(sum/(lenList[i]-lag))
        reMatrix[i] = np.hstack((ls1, ls2))
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
    pro_x = x.drop([0], axis=1).values
    pro_y = x[0].values
    pro_x = pd.DataFrame(pro_x)
    pro_x.to_csv('train_RPSSM1010.csv')
    pro_y = pd.DataFrame(pro_y)
    pro_y.to_csv('train_lable.csv')

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
    pro_px = x.drop([0], axis=1).values
    pro_py = x[0].values

    pro_px = pd.DataFrame(pro_px)
    pro_px.to_csv('test_RPSSM1010.csv')
    pro_py = pd.DataFrame(pro_py)
    pro_py.to_csv('test_lable.csv')


main()

