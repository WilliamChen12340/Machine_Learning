# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 22:48:02 2017

@author: williamchen12340
"""

import numpy as np
import random

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
        x_list = list[:4]
        x_list.append(1)
        X.append(np.array(x_list))
        y.append(list[4])
    file.close()
    return X , y

def error_counter(W,X,y):
    num=0
    for i in range(len(X)):
        if np.dot(W,X[i])*y[i] <= 0:
            num+=1
    return num

def Pocket(W, X ,y , eta , ud_num):
    ran_order = [x for x in range(len(X))]
    random.shuffle(ran_order)
    W_best = W
    ud_cnt = 0
    i = 0
    while 1:
        if ud_cnt == ud_num:
            break
        #分錯
        if np.dot(W,X[ran_order[i]])*y[ran_order[i]] <= 0:
            W_ = W + eta*y[ran_order[i]]*X[ran_order[i]]
            if error_counter(W_,X,y) < error_counter(W,X,y):
                W_best = W_
                ud_cnt += 1
            W = W_
        i=(i+1)%400
    return W_best
    
def Pocket_exp(W , X_train , y_train , X_test , y_test , eta , ud_num , exp_num):
    total_error_rate = 0
    for i in range(exp_num):
        W = Pocket(W, X_train ,y_train , eta , ud_num)
        cnt_error = error_counter(W , X_test ,y_test)
        error_rate = cnt_error / len(X_test) 
        total_error_rate += error_rate
        avg_rate = total_error_rate / exp_num
    return avg_rate

def PLA_random(W, X ,y , eta , ud_num):
    ran_order = [x for x in range(len(X))]
    random.shuffle(ran_order)
    cnt4adj = 0
    i = 0
    while 1:
        if cnt4adj == ud_num:
            break
        #分錯
        if np.dot(W,X[ran_order[i]])*y[ran_order[i]] <= 0:
            W = W + eta*y[ran_order[i]]*X[ran_order[i]]
            cnt4adj += 1
        i=(i+1)%400
    return W

def PLA_exp(W , X_train , y_train , X_test , y_test , eta , ud_num , exp_num):
    total_error_rate = 0
    for i in range(exp_num):
        W = PLA_random(W, X_train ,y_train , eta , ud_num)
        cnt_error = error_counter(W , X_test ,y_test)
        error_rate = cnt_error / len(X_test) 
        total_error_rate += error_rate
        avg_rate = total_error_rate / exp_num
    return avg_rate
    
W = np.array([0,0,0,0,0])        
X_train , y_train = read_in('hw1_18_train.dat.txt')
X_test , y_test = read_in('hw1_18_test.dat.txt')
print(Pocket_exp(W , X_train , y_train , X_test , y_test , 1 , 100 , 20))