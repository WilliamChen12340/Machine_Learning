# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:54:57 2017

@author: williamchen12340
"""

import numpy as np
import random
import matplotlib.pyplot as plt
#------------------------------------------------------------
def DecisionStump(X , y): #for 1-D
    Winner = len(X) + 1
    for i in (range(len(X)+1)):
        s_plus_cnt = 0
        s_minus_cnt = 0
        #pick theta
        if i > 0 and i < len(X):
            theta = (X[i-1] + X[i]) / 2
        elif i == 0:
            theta = (-1 + X[0]) / 2
        elif i == len(X):
            theta = (X[i-1] + 1) / 2
        #choose best s in the theta    
        for j in range(len(X)):
            if (X[j] < theta and y[j] > 0) or (X[j] > theta and y[j] < 0):
                s_plus_cnt += 1 
            if (X[j] < theta and y[j] < 0) or (X[j] > theta and y[j] > 0):
                s_minus_cnt += 1
        if s_plus_cnt < s_minus_cnt:
            if s_plus_cnt < Winner:
                s = 1
                Winner = s_plus_cnt
                WinnerE_in = s_plus_cnt / len(X)
                WinnerTheta = theta
        else:
            if s_minus_cnt < Winner:
                s = -1
                Winner = s_minus_cnt
                WinnerE_in = s_minus_cnt / len(X)
                WinnerTheta = theta
    E_out = (1/2) + (3/10)*s*(abs(WinnerTheta) - 1)
    return s , WinnerTheta , WinnerE_in , E_out
 #------------------------------------------------------------               
def DecisionStumpExp(n):
    EinList=[]
    EoutList=[]
    for i in range(n):
        X ,y = Generate()
        s , WinnerTheta , WinnerE_in , E_out = DecisionStump(X , y)
        EinList.append(WinnerE_in)
        EoutList.append(E_out)
    plt.scatter(EinList ,EoutList)
    avgE_in = np.mean(EinList)
    avgE_out = np.mean(EoutList)
    return avgE_in , avgE_out
#------------------------------------------------------------
def N_D_DecisionStump(X , y):
    AllWinner_Ein = ( len(X) + 1 ) / len(X) 
    for j in range(len(X[0])):
        X_ = []
        for i in range(len(X)):
            X_.append(X[i][j])
        s , WinnerTheta , Winner_Ein , E_out = DecisionStump(X_ , y)
        if Winner_Ein < AllWinner_Ein:
            AllWinner_Ein = Winner_Ein
            AllWinner_s = s
            AllWinnerTheta = WinnerTheta
    return AllWinner_s , AllWinnerTheta , AllWinner_Ein
#------------------------------------------------------------
def N_D_DecisionStumpTest(X_train , y_train, X_test , y_test):
    AllWinner_s , AllWinnerTheta , AllWinnerE_in = N_D_DecisionStump(X_train , y_train)
    AllWinner_Eout = ( len(X_test) + 1 ) / len(X_test)
    for j in range(len(X_test[0])):
        X_ = []
        Eout_cnt = 0
        for i in range(len(X_test)):
            X_.append(X_test[i][j])
            if ((X_[i] - AllWinnerTheta) * AllWinner_s)*(y_test[i]) < 0:
                Eout_cnt += 1
        Eout = Eout_cnt / len(X_test)
        if Eout < AllWinner_Eout:
            AllWinner_Eout = Eout
    return AllWinner_Eout
#------------------------------------------------------------
def Generate(): #generate data
    X = np.random.uniform(-1,1,20)
    y = []
    X.sort()
    for i in range(len(X)):
        rand = random.randint(0,4)
        if rand == 0:
            if X[i] < 0: 
                y.append(1)
            else:
                y.append(-1)
        else:
            if X[i] < 0:
                y.append(-1)
            else:
                y.append(1)
    return X ,y
#------------------------------------------------------------
def read_in(file_name):
    X = []
    y = []
    file = open(file_name , 'r')
    while 1:    
        line = file.readline()
        if not line:
           break
        list = line.strip().split()
        for i in range(len(list)):
            list[i] = float(list[i])
        x_list = list[:9]
        X.append(np.array(x_list))
        y.append(list[9])
    file.close()
    return X , y
#------------------------------------------------------------
X_train , y_train = read_in('hw2_train.dat.txt')
X_test , y_test = read_in('hw2_test.dat.txt')
print(DecisionStumpExp(1000))