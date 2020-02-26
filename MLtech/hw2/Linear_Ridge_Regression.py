#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:47:23 2018

@author: williamchen12340
"""
import time
import numpy as np
import csv
np.random.seed(0)
A=[]
a=[]
x=[]
y=[]
with open('/Users/williamchen12340/MLThw2/hw2_lssvm_all.dat.csv', newline='') as data:
    rows = csv.reader(data)
    for row in rows:
        A.append(list(map(float ,row[0].split())))
    for i in range(len(A)):
        #y.append(A[i][0])
        for j in range(len(A[0])-1):
            a.append(A[i][j])
        y.append(A[i][len(A[0])-1])
        x.append(a)
        a = []
# split train and test
x_train = x[:400]
y_train = y[:400]
x_test = x[400:]
y_test = y[400:]
#%%
def guassian_rbf(gam, xm, xn):
    xm = np.array(xm)
    xn = np.array(xn)
    diff = xm - xn
    result = np.exp(-gam * np.dot(diff, diff))
    return result
#%%
def kernel_matrix(gam, x):
    N = len(x)
    K = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            K[i,j] = guassian_rbf(gam, x[i], x[j])
    return K
#%%
def kernel_ridge_regression(x_train, y_train, x_test, y_test, lamda, gamma ):
    print('gamma\tlambda\tEin\tEout')
    for gam in gamma:
        for lam in lamda:
            #print(gam)
            #print('check point0')
            K = kernel_matrix(gam, x_train)
            beta = np.dot(np.linalg.inv(lam * np.eye(len(x_train)) + K), y_train)
            #print('check point1')
            Ein = 0
            for i in range(len(x_train)):
                kernel = np.array([guassian_rbf(gam, x_train[j], x_train[i]) for j in range(len(x_train))])
                predict = np.sum(beta * kernel)
                if predict * y_train[i] <= 0:
                    Ein += 1 
            Eout = 0
            for i in range(len(x_test)):
                kernel = np.array([guassian_rbf(gam, x_train[j], x_test[i]) for j in range(len(x_train))])
                predict = np.sum(beta * kernel)
                if predict * y_test[i] <= 0:
                    Eout += 1 
            print("%g\t%g\t%g\t%g" % (gam, lam, Ein / len(x_train), Eout / len(x_test)))

#%% prob 11&12        
lamda = [1e-3, 1, 1e3]
gamma = [32, 2, 0.125]
kernel_ridge_regression(x_train, y_train, x_test, y_test, lamda, gamma )
#%%
def LinKernelMatrix(x):
    K = np.dot(np.array(x),np.array(x).T)
    '''
    N = len(x)
    K = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            K[i,j] = np.dot(x[i], x[j])
    '''
    return K
#%%
def Lin_ridge_regression(x_train, y_train, x_test, y_test, lamda):
    for lam in lamda:
            K = LinKernelMatrix(x_train)
            beta = np.dot(np.linalg.inv(lam * np.eye(len(x_train)) + K), y_train)
            w = np.dot(beta, x_train)
            Ein = 0
            for i in range(len(x_train)):
                #kernel = np.array([np.dot(x_train[j], x_train[i]) for j in range(len(x_train))])
                predict = np.dot(w, x_train[i])
                if predict * y_train[i] <= 0:
                    Ein += 1 
            Eout = 0
            for i in range(len(x_test)):
                #kernel = np.array([np.dot(x_train[j], x_test[i]) for j in range(len(x_train))])
                predict = np.dot(w, x_test[i])
                if predict * y_test[i] <= 0:
                    Eout += 1 
            print("%g\t%g\t%g" % (lam, Ein / len(x_train), Eout / len(x_test)))
    return beta
#%% prob 13&14
for xi in x_train:
    xi.append(1)
for xi in x_test:
    xi.append(1)
lamda = [0.01, 0.1, 1, 10, 100]
print('-------prob13&14--------')
print('lambda\tEin\tEout')
Lin_ridge_regression(x_train, y_train, x_test, y_test, lamda)
#%%
def booststrap(x_train, ytrain, n):
    xd_t = []
    yd_t = []
    idx = np.random.randint(0, n, n)
    for i in idx:
        xd_t.append(x_train[i])
        yd_t.append(y_train[i])
    return xd_t, yd_t
#%%
def Lin_ridge_regression_rep(x_train, y_train, lamda):
    for lam in lamda:
            K = np.dot(np.array(x_train), np.array(x_train).T)
            beta = np.dot(np.linalg.inv(lam * np.eye(len(x_train)) + K), y_train)
    return beta
#%%
def Bagging_Lin_ridge_regression(x_train, y_train, x_test, y_test, lamda):
    for lam in lamda:
        #W = np.array(0)
        w = []
        for j in range(250):
            xd_t = []
            yd_t = []
            xd_t, yd_t = booststrap(x_train, y_train, 400)
            beta = Lin_ridge_regression_rep(xd_t, yd_t, [lam])
            w.append(np.dot(beta, xd_t))
            
        Ein = 0
        for i in range(len(x_train)):
            predict = np.sum(np.sign(np.dot(w, x_train[i])))
            if predict * y_train[i] <= 0:
                Ein += 1
        Eout = 0
        for i in range(len(x_test)):
            predict = np.sum(np.sign(np.dot(w, x_test[i])))
            if predict * y_test[i] <= 0:
                Eout += 1
        print("%g\t%g\t%g" % (lam, Ein / len(x_train), Eout / len(x_test)))
#%% prob 15&16
lamda = [0.01, 0.1, 1, 10, 100]
print('-------prob 15&16-------')
print('lambda\tEin\tEout')
start = time.time()
Bagging_Lin_ridge_regression(x_train, y_train, x_test, y_test, lamda)
end = time.time()
print(end-start)