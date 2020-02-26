# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 09:43:41 2018

@author: williamchen12340
"""
import numpy as np
import matplotlib.pyplot as plt
import time

train_data = 'hw3_train.dat.txt'
test_data = 'hw3_test.dat.txt'

#%%
def readData(data):
    X = []
    Y = []
    with open(data, 'r') as file:
        for line in file:
            data = line.split()
            tmp = float(data[0]), float(data[1])
            X.append(tmp)
            Y.append(int(data[-1]))
    return np.array(X), np.array(Y)

#%%
def all_positive(array):
    for i, ele in enumerate(array):
        if ele < 0:
            array[i] = ele* -1
    return array

#%%
def update_U(scal, U, E_in):
    for i, u in enumerate(U):
        if E_in[i]:
            U[i] = u*scal
        else:
            U[i] = u/scal
    return U

#%%
def G_error(G, alpha, X, Y):
    ans = []
    for x in X:
        consensus = 0
        for (i, s, theta), a in zip(G, alpha):
            consensus += a*(np.sign(s*(x[i] - theta)))
        ans.append(np.sign(consensus))
    error = all_positive(ans - Y)/2
    return error
    
#%% 
def theta(X, idx, Y, U):
    best_theta, best_s, E_in = 0, 1, np.array([np.inf]*len(X))
    for i, index in enumerate(idx):
        if i+1 < len(idx):
            tmp_theta = (X[idx[i]]+ X[idx[i+1]])/2  #each position for theta
            
            left_nr = len([x for x in X if x < tmp_theta])  #less than theta
            right_nr = len(X) -left_nr                         #bigger than theta
            left = np.array([-1] * left_nr)     #left y     i.e. -1
            right = np.array([1] * right_nr)    #right y    i.e. +1
            pred_y = np.concatenate((left, right))  #total predict y
            neg_y = pred_y *-1                  #s = -1
            
            tmp_E_in = all_positive(pred_y - Y)*0.5  #when predict != lebal
            neg_E_in = all_positive(neg_y - Y)*0.5   #s = -1 situation
            tmp_s = 1
            if np.dot(neg_E_in, U) < np.dot(tmp_E_in, U):
                tmp_s = -1
                tmp_E_in = neg_E_in 
            
            if np.dot(tmp_E_in, U) < np.dot(E_in, U) :
                best_theta = tmp_theta
                E_in = tmp_E_in
                best_s = tmp_s
    
    #all '1' & all '-1'                
    tmp_theta = -np.inf
    tmp_s = 1
    nr = len(X)
    pridct_y = np.array([1] *nr)
    tmp_E_in = all_positive(pridct_y - Y)*0.5
    neg_y = pridct_y * -1
    neg_E_in = all_positive(neg_y - Y)*-0.5
    if np.dot(neg_E_in, U) < np.dot(tmp_E_in, U):
        tmp_s = -1
    if np.dot(tmp_E_in, U) < np.dot(E_in, U):
        best_theta = tmp_theta
        E_in = tmp_E_in
        best_s = tmp_s
                
    return best_theta, best_s, E_in
#%%
def predict(g, a, X):
    ans = []
    d, s, theta = g
    for x in X:
        score = a*np.sign(s*(x[d] - theta))
        ans.append(score)
    return np.array(ans)

#%%        
def cal_alpha(E_in, U):
    epsilon = np.dot(E_in, U)/sum(U)
    scaling = np.sqrt((1-epsilon)/ epsilon)
    ans = np.log(scaling)
    return ans, scaling

#%%
def hw_3(data):
    start = time.time()
    X, Y = readData(data)
    X_1D = np.array([x[0] for x in X])
    sorted_X1D_idx = np.argsort(X_1D)
    D1Y = Y[sorted_X1D_idx]
    D1_restore = np.argsort(sorted_X1D_idx)
    X_2D = np.array([x[1] for x in X])
    sorted_X2D_idx = np.argsort(X_2D)
    D2Y = Y[sorted_X2D_idx]
    D2_restore = np.argsort(sorted_X2D_idx)
    check_u = []
    N = len(X_1D)
    U = np.array([1/N] *N)
    all_U = []
    T = 300
    G = []
    A = []
    best_theta, best_s = 0, 1
    Ein_g =[]
    u = 0
    for i in range(T):
        check_u.append(U)
#        print('\rIteration: %d'%(i+1), end = '',flush = True)
        #finding gt
        D1U = U[sorted_X1D_idx]
        D2U = U[sorted_X2D_idx]
        D1_best_theta, D1_best_s, D1_Ein = theta(X_1D, sorted_X1D_idx, D1Y, D1U)
        D2_best_theta, D2_best_s, D2_Ein = theta(X_2D, sorted_X2D_idx, D2Y, D2U)
        if np.dot(D1_Ein, D1U) < np.dot(D2_Ein, D2U):
            best_theta, best_s = D1_best_theta, D1_best_s
            g = (0, best_s, best_theta)
            E_in = D1_Ein
            u = 1
            U = D1U
        else:
            best_theta, best_s = D2_best_theta, D2_best_s
            g = (1, best_s, best_theta)
            E_in = D2_Ein
            u = 0
            U = D2U
        alpha, scal = cal_alpha(E_in, U)
        all_U.append(sum(U))
        U = update_U(scal, U, E_in)
        
        if u == 1:
            U = U[D1_restore]
        else:
            U = U[D2_restore]
            
        Ein_g.append(sum(E_in)/N)
        A.append(alpha)
        G.append(g)
    end = time.time()
    print('running time:%d' %(end - start))
    return G, A, all_U, Ein_g

#%% 
X_test, Y_test = readData(test_data)
X, Y = readData(train_data)
print('\nQuestion 11')
G, A, all_U, Ein_g = hw_3(train_data)
print('\ng1 error: {}'.format(Ein_g[0]))
print('\nalpha_1: {}'.format(A[0]))
plt.figure()
plt.plot(Ein_g, 'b')
plt.xlabel('t')
plt.ylabel('0/1 error')
plt.title('Ein(gt)')
plt.show()

#%%
print('\nQuestion 13')
Ein_G = []
score = np.zeros(len(X))
for g, a in zip(G, A):
    score = score + predict(g, a, X)
    error = all_positive(np.sign(score) - Y)/2
    Ein_G.append(sum(error)/len(X))
print('\n Ein_G: {}'.format(Ein_G[-1]))
plt.figure()
plt.plot(Ein_G, 'b')
plt.show()

#%%
print('\nQuestion 14')
print('\nU_2: {}'.format(all_U[1]))
print('\nUt: {}'.format(all_U[-1]))
plt.figure()
plt.plot(all_U, 'b')
plt.xlabel('t')
plt.ylabel('0/1 error')
plt.title('U')
plt.show()

#%%
print('\nQuestion 15')
Eout_g = []


for (d, s, th) in G:
    error = 0
    for i, x in enumerate(X_test):    
        _predict = np.sign(s*(x[d]- th))
        if _predict != Y_test[i]:
            error += 1
    Eout_g.append(error/len(X_test))
    
print('Eout_g_1: {}'.format(Eout_g[0]))
plt.figure()
plt.plot(Eout_g, 'b')
plt.xlabel('t')
plt.ylabel('0/1 error')
plt.title('Eout(gt)')
plt.show()

#%%
print('\nQuestion 16')
Eout_G = []
score = np.zeros(len(X_test))
for g, a in zip(G, A):
    score = score + predict(g, a, X_test)
    error = all_positive(np.sign(score) - Y_test)/2
    Eout_G.append(sum(error)/len(X_test))

print('\nEout_Gt: {}'.format(Eout_G[-1]))

plt.figure()
plt.plot(Eout_G, 'b')
plt.xlabel('t')
plt.ylabel('0/1 error')
plt.title('Eout(Gt)')
plt.show()
