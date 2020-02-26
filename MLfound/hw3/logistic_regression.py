# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 16:52:56 2017

@author: williamchen12340

logistic regrssion optimized by gradient descent
"""

import numpy as np
import matplotlib.pyplot as plt 

f = open('hw3_train.dat.txt','r')
data = []
for line in f:
    l = line.split()
    data.append([np.concatenate((np.array([1]),[float(i) for i in l[:-1]])),int(l[-1])])
f.close()

f = open('hw3_test.dat.txt','r')
testing = []
for line in f:
    l = line.split()
    testing.append([np.concatenate((np.array([1]),[float(i) for i in l[:-1]])),int(l[-1])])
f.close()


def sigmoid(z):
    return 1/(1+ np.exp(-1*z))

def gradient(dataset, w):
    g = np.zeros(len(w))
    for i in range(len(dataset)):
        x = dataset[i][0]
        y = dataset[i][1] 
        theta = sigmoid((-1)*y* (w.T.dot(x)))
        g += (-1) * y *theta* x
    return g / len(dataset)

def SGD(dataset,w,i):
    x = dataset[i][0]
    y = dataset[i][1] 
    theta = sigmoid((-1)*y* (w.T.dot(x)))
    g = (-1) * y *theta* x
    return g
    
def cost(dataset,w):
    total_cost = 0
    for i in range(len(dataset)):
        x = dataset[i][0]
        y = dataset[i][1] 
        if y*(w.T.dot(x)) <= 0:
            total_cost += 1
    return total_cost / float(len(dataset))

def logistic_GD_Ein(train, eta, T):
    w = np.zeros(21)
    Ein = []
    for i in range(T):
        current_cost = cost(train, w)
        Ein.append(current_cost)
        w = w - eta* gradient(train, w)
    return Ein

def logistic_SGD_Ein(train, eta, T):
    w = np.zeros(21)
    Ein = []
    for i in range(T):
        current_cost = cost(train, w)
        Ein.append(current_cost)
        w = w - eta* SGD(train, w , i%len(train))
    return Ein

def logistic_GD_Eout(train,test, eta, T):
    w = np.zeros(21)
    Eout = []
    for i in range(T):
        current_cost = cost(test, w)
        Eout.append(current_cost)
        w = w - eta* gradient(train, w)
    return Eout

def logistic_SGD_Eout(train,test, eta, T):
    w = np.zeros(21)
    Eout = []
    for i in range(T):
        current_cost = cost(test, w)
        Eout.append(current_cost)
        w = w - eta* SGD(train, w , i%len(train))
    return Eout

x = np.arange(2000)

plt.plot(x , logistic_GD_Eout(data,testing ,0.001, 2000))
plt.plot(x , logistic_GD_Eout(data,testing ,0.01, 2000))
plt.plot(x , logistic_SGD_Eout(data,testing ,0.001, 2000))
plt.show()


