#!/usr/bin/env python
# coding: utf-8

#To Study Various Type of Activation Function in Neural Network.

#  # Importing Libraries

# In[101]:


from numpy import array #For Array Initialization 
from numpy import random #For Randomly Choosing Numbers 
from numpy import dot #For Doing Dot Product
from random import choice


# # Initialized Dataset

# In[102]:


dataset = [
    (array([0,0,1]),0),# array([x,y,b],e) x,y=input, b=bias, e=Expected O/P to validate
    (array([0,1,1]),1),
    (array([1,0,1]),1),
    (array([1,1,1]),1),
]
print(dataset)
print(array([0,0,1]),0)


# # Initializing Random numbers for Weights

# In[103]:


weights = random.rand(3)
weights


# # Initializing Additional Varaiables

# In[104]:


r = 0.2 #Learning Rate
n = 100 #Number of Iteration


# # Activation Funtion

# # STEP Activation Function

# In[105]:


activationFn = lambda x: 1 if x > 0 else 0 #step activation function (if i/p is negative o/p is 0 else 1)

for j in range(n): 
    x, expected = choice(dataset)
    result = dot(weights, x)
    err = expected-activationFn(result)
    weights += r * err * x

for x, _ in dataset:
    result = dot(x, weights)  
    print("ResultBAFn: {} ResultAFn {}".format(round(result,4), activationFn(result)))                     


# # Linear Activation Function

# In[106]:


activationFn = lambda x: x

for j in range(n): 
    x, expected = choice(dataset)
    result = dot(weights, x)
    err = expected-activationFn(result)
    weights += r * err * x

for x, _ in dataset:
    result = dot(x, weights)  
    print("ResultBAFn: {} ResultAFn {}".format(round(result,3), activationFn(result)))     


# # Sigmoid Activation Function

# In[107]:


import numpy as np
activationFn = lambda x: 1/(1+np.exp(-x))

for j in range(n): 
    x, expected = choice(dataset)
    result = dot(weights, x)
    err = expected-activationFn(result)
    weights += r * err * x

for x, _ in dataset:
    result = dot(x, weights)  
    print("ResultBAFn: {} ResultAFn {}".format(round(result,3), activationFn(result)))     


# # RELU Activation Function

# In[108]:


activationFn = lambda x: x if x > 0 else 0

for j in range(n): 
    x, expected = choice(dataset)
    result = dot(weights, x)
    err = expected-activationFn(result)
    weights += r * err *x 
    

for x, _ in dataset:
    result = dot(x, weights)  
    print("ResultBAFn: {} ResultAFn {}".format(round(result,3), activationFn(round(result,3))))     


# # SOFTMAX Activation function

# In[109]:


activationFn = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
err = []
for j in range(n): 
    x, expected = choice(dataset)
    result = dot(weights, x)
    err = expected-activationFn(result)
    weights += r * err * x

for x, _ in dataset:
    result = dot(x, weights)  
    print("ResultBAFn: {} ResultAFn {}".format(round(result,3) , activationFn(result)))     

