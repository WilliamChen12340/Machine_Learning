#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 03:50:02 2018

@author: williamchen12340
"""
#import numpy as np
import math
from cvxopt import matrix, solvers
#%% creat data
x = [[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]]
y = [-1,-1,-1,1,1,1,1]
z = []
for i in range(len(x)):
    z1 = 2*math.pow(x[i][1] ,2) - 4*x[i][0] + 2
    z2 = math.pow(x[i][0] ,2) - 2*x[i][1] - 1
    z.append([z1,z2])
#%% solving QP for prob1
Q_t = matrix([[0.0 ,0.0 ,0.0] ,[0.0 ,1.0 ,0.0] ,[0.0 ,0.0 ,1.0]])
Q = Q_t.trans()
p = matrix([0.0 ,0.0 ,0.0])
G_t = matrix([[-y[0] * matrix([1.0 ,z[0][0] ,z[0][1]])] ,[-y[1] * matrix([1.0 ,z[1][0] ,z[1][1]])] 
             ,[-y[2] * matrix([1.0 ,z[2][0] ,z[2][1]])] ,[-y[3] * matrix([1.0 ,z[3][0] ,z[3][1]])]
             ,[-y[4] * matrix([1.0 ,z[4][0] ,z[4][1]])] ,[-y[5] * matrix([1.0 ,z[5][0] ,z[5][1]])]
             ,[-y[6] * matrix([1.0 ,z[6][0] ,z[6][1]])]])
G = G_t.trans()
h = matrix([-1.0 ,-1.0 ,-1.0 ,-1.0 ,-1.0 ,-1.0 ,-1.0])
sol_1 = solvers.qp(Q ,p ,G ,h)
print(sol_1['x'])
#%% solving QP for prob2
q = []
a = []
for i in range(len(x)):
    for j in range(len(x)):
        a.append(y[i] * y[j] * (1 + 2 * math.pow((x[i][0] * x[j][0] + x[i][1] * x[j][1]) ,2)))
    q.append(a)
    a = []
Q_t = matrix([q[0] ,q[1] ,q[2] ,q[3] ,q[4] ,q[5] ,q[6]])
Q = Q_t.trans()
p = matrix([-1.0]*7)
G = matrix([[-1.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0] ,[0.0 ,-1.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0] 
            ,[0.0 ,0.0 ,-1.0 ,0.0 ,0.0 ,0.0 ,0.0] ,[0.0 ,0.0 ,0.0 ,-1.0 ,0.0 ,0.0 ,0.0]
            ,[0.0 ,0.0 ,0.0 ,0.0 ,-1.0 ,0.0 ,0.0] ,[0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,-1.0 ,0.0]
            ,[0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,-1.0]])
h = matrix([0.0]*7)
A = matrix([-1.0,-1.0,-1.0,1.0,1.0,1.0,1.0], (1,7))
b = matrix(0.0)
sol_2= solvers.qp(Q ,p ,G ,h ,A ,b)
print(sol_2['x'])
