# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 23:05:58 2017

@author: williamchen12340
"""
import numpy as np
import random
from matplotlib import pyplot as plt 
#讀入資料
X = []
y = []
Z = []
file = open('hw1_15_train.dat.txt' , 'r')
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


#PLA_in_order
def PLA_in_order(W , X , y):
    cnt4end = 0
    cnt4adj = 0
    i = 0
    while 1:
        if cnt4end == 400:
            break
        #分錯
        if np.dot(W,X[i])*y[i] < 0 or ( y[i] == 1 and np.dot(W,X[i]) == 0 ):
            W = W + y[i]*X[i]
            cnt4end = 0
            cnt4adj += 1
        else:
            cnt4end += 1
        i=(i+1)%400
    return cnt4adj

#PLA_random
def PLA_random(W, X ,y , eta):
    ran_order = [x for x in range(len(X))]
    random.shuffle(ran_order)
    cnt4end = 0
    cnt4adj = 0
    i = 0
    while 1:
        if cnt4end == 400:
            break
            #分錯
        if np.dot(W,X[ran_order[i]])*y[ran_order[i]] < 0 or ( y[ran_order[i]] == 1 and np.dot(W,X[ran_order[i]]) == 0 ):
            W = W + eta*y[ran_order[i]]*X[ran_order[i]]
            cnt4end = 0
            cnt4adj += 1
        else:
            cnt4end += 1
        i=(i+1)%400
    return cnt4adj
#PLA_random_experiment    
def PLA_ran_exp(W , X , y , eta , exp_num):
    total_cnt = 0
    for i in range(exp_num):
        update_num = PLA_random(W , X , y , eta)
        Z.append(update_num)
        total_cnt += update_num
    return total_cnt / exp_num  
  
W = np.array([0, 0, 0, 0, 0])
print(PLA_ran_exp(W , X , y , 1 , 2000))
np.array(Z)
print(Z)

bins = []
for i in range(101):
    bins.append(i)

print(bins)
plt.xlabel("number of update")
plt.ylabel("freq of number")
plt.hist(Z, bins)
plt.show()







