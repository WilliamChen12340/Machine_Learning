#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 01:32:42 2018

@author: williamchen12340

SVM
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import math
import csv
import os
os.chdir('/Users/williamchen12340/libsvm-3.22/python')
from svmutil import *
#%%
def prob11(y):
    label = []
    for i in range(len(y)):
        if y[i] != 0:
            label.append(-1)
        else:
            label.append(1)
    return label
#%%
def prob12(y):
    label = []
    for i in range(len(y)):
        if y[i] != 8:
            label.append(-1)
        else:
            label.append(1)
    return label
#%%
def get_w(z ,m):
    w = np.zeros(len(z[0]))
    SV_coef = m.get_sv_coef()
    SV_id = m.get_sv_indices()
    for i in range(len(SV_id)):
        w = w + SV_coef[i] * np.array(z[SV_id[i] - 1])
    return w
#%%
def w_norm(w):
    a = np.dot(w,w)
    return a**0.5
#%%
def get_w_norm(x ,m):
    w_sq = 0
    SV_coef = m.get_sv_coef()
    SV_id = m.get_sv_indices()
    for i in range(len(SV_coef)):
        for j in range(len(SV_coef)):
            diff = np.array(x[SV_id[i] - 1]) - np.array(x[SV_id[j] - 1])
            p = -80 * (np.dot(diff ,diff))
            w_sq = w_sq + SV_coef[i][0] * SV_coef[j][0] * np.exp(p)
    result = math.pow(w_sq ,0.5)
    return result
#%%
def get_freeSV_p_v_id(m ,c):
    i = 0
    SV_coef = m.get_sv_coef()
    SV_id = m.get_sv_indices()
    if SV_coef[i][0] == c:
        while SV_coef[i][0] == c or SV_coef[i][0] == -c:
            i = i + 1
            if SV_coef[i][0] != c and SV_coef[i][0] != -c:
                break
    else:
        i=0
    idx = SV_id[i] - 1
    return idx
#%%
def random_idx(total ,n):
    ran_idx = random.sample(range(total), n)
    return ran_idx
#%%
A = [] 
y = []
x = []
yt= []
xt= []
a = []
#read train data
with open('/Users/williamchen12340/MLThw1/features.train.csv' ,newline='') as train_data:
    rows = csv.reader(train_data)
    for row in rows:
        A.append(list(map(float ,row[0].split())))
    for i in range(len(A)):
        y.append(A[i][0])
        for j in range(len(A[0])-1):
            a.append(A[i][j+1])
        x.append(a)
        a = []
#read test data
A = []
with open('/Users/williamchen12340/MLThw1/features.test.csv' ,newline='') as test_data:
    rows = csv.reader(test_data)
    for row in rows:
        A.append(list(map(float ,row[0].split())))
    for i in range(len(A)):
        yt.append(A[i][0])
        for j in range(len(A[0])-1):
            a.append(A[i][j+1])
        xt.append(a)
        a = []
        
#%%prob11
c = [10**-5, 10**-3, 10**-1, 10, 10**3]
y_label = []
SV_coef = []
SV_id = []
W_norm_all = []
for i in range(len(c)):
    print('running for C{}'.format(i))
    y_label = prob11(y)
    prob = svm_problem(y_label ,x)
    para = svm_parameter('-t 0 -h 0 -c {}'.format(c[i]))
    m = svm_train(prob ,para)
    W_norm_all.append(w_norm(get_w(x ,m)))   
    #svm_save_model('model4prob11_c{}'.format(i) ,m)
#plot
plt.figure(1)
plt.scatter([-5,-3,-1,1,3] ,W_norm_all)
plt.plot([-5,-3,-1,1,3] ,W_norm_all)
plt.ylabel('||W||')
plt.title('Problem 11')
plt.xlabel('log C')
plt.show()

#%%prob12 & 13
c = [10**-5, 10**-3, 10**-1, 10, 10**3]
Ein_all = []
SV_num = []
y_label = prob12(y) 
for i in range(len(c)):
    prob = svm_problem(y_label ,x)
    para = svm_parameter('-t 1 -d 2 -g 1 -r 1 -c {}'.format(c[i]))
    m = svm_train(prob ,para)
    p_l, p_acc, p_v = svm_predict(y_label, x, m)
    Ein_all.append(1 - p_acc[0]/100)
    #svm_save_model('model4prob12and13_c{}'.format(i) ,m)
    SV_coef = m.get_sv_coef()
    SV_id = m.get_sv_indices()
    SV_num.append(len(SV_id))
#plot12
plt.figure(2)
plt.scatter([-5,-3,-1,1,3],Ein_all)
plt.plot([-5,-3,-1,1,3],Ein_all)
plt.ylabel('Ein')
plt.title('problem 12')
plt.xlabel('log C')
plt.show()    
#plot13
plt.figure(3)
plt.scatter([-5,-3,-1,1,3],SV_num)
plt.plot([-5,-3,-1,1,3],SV_num)
plt.xlabel('log10 C')
plt.ylabel('nr_sv')
plt.title('Prooblem 13')
plt.show()

#%%prob14
c = [10**-3, 10**-2, 10**-1, 1, 10]
dis_all = []
for i in range(len(c)):
    print('running for C{}'.format(i))
    y_label = prob11(y)
    prob = svm_problem(y_label ,x)
    para = svm_parameter('-t 2 -g 80 -c {}'.format(c[i]))
    m = svm_train(prob ,para)    
    #svm_save_model('model4prob14_c{}'.format(i) ,m)
    p_l, p_acc, p_v = svm_predict(y_label, x, m)
    idx = get_freeSV_p_v_id(m ,c[i])
    p_freeSV = p_v[idx]
    w_norm_ = get_w_norm(x ,m)
    dis = abs(p_freeSV[0]) / w_norm_
    dis_all.append(dis)
    SV_coef = m.get_sv_coef()
    SV_id = m.get_sv_indices()
#plot14  
plt.figure(4)
plt.scatter([-3,-2,-1,0,1],dis_all)
plt.plot([-3,-2,-1,0,1],dis_all)
plt.xlabel('log10 C')
plt.ylabel('distance from free')
plt.title('problem 14')
plt.show()


#%%prob15
y_label = prob11(y) 
yt_label = prob11(yt)
g = [1, 10, 10**2, 10**3, 10**4]
Eout_all = []
for i in range(len(g)):
    prob = svm_problem(y_label ,x)
    para = svm_parameter('-t 2 -c 0.1 -g {}'.format(g[i]))
    m = svm_train(prob ,para)    
    #svm_save_model('model4prob15_g{}'.format(i) ,m)
    #p_l, p_acc, p_v = svm_predict(y_label ,x ,m)
    p_l, p_acc, p_v = svm_predict(yt_label ,xt ,m)
    Eout_all.append(1- p_acc[0]/100)
#plot15
plt.figure(5)
plt.scatter([0,1,2,3,4],Eout_all)
plt.plot([0,1,2,3,4],Eout_all)
plt.ylabel('E_out')
plt.title('problem 15')
plt.xlabel('log gamma')
plt.show()   
#%%prob16
y_label = prob11(y)
g = [10**-1 ,1 ,10 ,10**2 ,10**3]
choose = []
for l in range(100):
    print('{}th round'.format(l+1))
    Eval_all = []
    x_val = []
    y_label_val = []
    y_temp = y_label[:]
    x_temp = x[:]    
    #ran_idx = random_idx(len(x) ,100)
    for j in range(1000):
        idx = random.sample(range(len(x_temp)), 1)
        x_val.append(x_temp[idx[0]])
        y_label_val.append(y_temp[idx[0]])
        del x_temp[idx[0]]
        del y_temp[idx[0]]
    for i in range(len(g)):
        prob = svm_problem(y_temp ,x_temp)
        para = svm_parameter('-t 2 -c 0.1 -g {}'.format(g[i]))
        m = svm_train(prob ,para)    
        p_l, p_acc, p_v = svm_predict(y_label_val ,x_val ,m)
        Eval_all.append(1- p_acc[0]/100)
    temp = Eval_all.index(min(Eval_all)) - 1
    choose.append(temp)                          
#plot16
plt.figure(6)
plt.hist(choose ,bins=[-1.5 ,-0.5 ,0.5 ,1.5 ,2.5 ,3.5])
plt.ylabel('gamma choice')
plt.title('problem 16')
plt.xlabel('log gamma')
plt.show()   
