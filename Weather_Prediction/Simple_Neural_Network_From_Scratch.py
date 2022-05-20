#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:17:08 2019

@author: noname797
"""

import numpy as np

#loading dataset
x=np.random.random(size=[3,4])
y=np.random.randint(0,2,size=[3,1])

#setting functions
def sigmoid(x):
    return 1/(1+np.exp(-x))
def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

#setting layers
input_layer=np.append(np.ones([3,1]),x,axis=1)
weights=np.random.random([5,1]) #w[0]=bias
output=sigmoid(np.dot(input_layer,weights))
error=y-output
adjustments=error*d_sigmoid(np.dot(input_layer,weights))
training_epochs=20000

for i in range(training_epochs+1):
    output=sigmoid(np.dot(input_layer,weights))
    error=y-output
    adjustments=error*d_sigmoid(np.dot(input_layer,weights))
    weights+=np.dot(input_layer.T,adjustments)
    if((i+1)%1000==1):
        print ("Training epoch = "+ str(i) +" Weights = "+ str(weights[1:])+" bias = "+ str(weights[0]))
        print("Predicted output ="+ str(output) +" Actual Output = "+str(y))
        
        
print('After '+ str(training_epochs)+ '  iterations..')
print("Final Weights = "+ str(weights[1:]) +" Final Bias = "+str(weights[0]))
