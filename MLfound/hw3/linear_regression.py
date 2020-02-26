# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 13:14:39 2017

@author: williamchen12340

feature transformation and linear regression
"""
import random
import numpy as np
import matplotlib.pyplot as plt
def func(x):
    x_1 = x[1]
    x_2 = x[2]
    ans =  np.sign(x_1**2 + x_2**2 - 0.6 )
    return ans

def sample_gen(num):
    samples = []
    for i in range(num):
        samples.append([1, np.random.uniform(-1,1), np.random.uniform(-1,1)])
    return samples


def label_gen(samples):
    flip = random.sample(range(1,len(samples)), 100)
    label = []    
    for i in range(len(samples)):
        label.append(func(samples[i]))
    for i in flip:
        if label[i] > 0:
            label[i] = -1.0
        else:
            label[i] = 1.0
    return label
    
def f_transform(samples):
    n_sample = []    
    for sample in samples:
        x1 = sample[1]
        x2 = sample[2]
        n_sample.append([1, x1, x2, x1*x2, x1**2, x2**2])
    return n_sample
    
def linear_regression(samples, label):
    w_s = np.linalg.pinv(samples)
    w = np.dot(w_s, label)
    return w

        
def problem_15(iteration, num):
    #total_err = 0
    samples = sample_gen(num)
    f_sample = f_transform(samples)
    label = label_gen(samples)
    w = linear_regression(f_sample, label)
    error = []
    for i in range(iteration):
        sample_out = sample_gen(num)
        sample_out_f = f_transform(sample_out)
        label_out = label_gen(sample_out)
        #start to verify
        err = 0
        for i in range(len(samples)):
            predict = np.sign(np.dot(w,sample_out_f[i]))
            if predict != label_out[i] :
                err += 1
        error.append(err/float(num))
        #total_err += err/float(num)
    #avg_err = total_err / float(iteration)
    return error
plt.hist(problem_15(1000,1000))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    