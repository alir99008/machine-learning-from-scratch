# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 10:29:45 2023

@author: Hp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading data file
df = pd.read_csv("house data.csv")

data_sample = df.iloc[: , :-1].values
label = df.iloc[: , 1].values

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(data_sample , label , test_size=0.3 , random_state=0)

learning_rate = 0.01

data , features = x_train.shape
weight = np.zeros((features,features))
bais = 0   


for i in range(600):
    
    mx = np.zeros((x_train.shape[0] , x_train.shape[1]))
    for j in range(x_train.shape[0]):
        for k in range(weight.shape[1]):
            total = 0
            for l in range(x_train.shape[1]):
                mul = x_train[j,l] * weight[l,k]
                total = total+mul
            mx[j,k] = total
    y_predicted = bais+mx
    
    
    chng = (y_predicted.T)-(y_train.T)
    chng = chng.T
    chng_weight = np.zeros((x_train.shape[0] , x_train.shape[1]))
    
    for j in range(x_train.shape[0]):
        for k in range(chng.shape[1]):
            total = 0
            for l in range(x_train.shape[1]):
                mul = x_train[j,l] * chng[l,k]
                total = total+mul
            chng_weight[j,k] = total
    new_chng_weight = (1/data)*chng_weight
    new_chng_bais = (1/data)*np.sum(chng)
    
    weight = weight - learning_rate * new_chng_weight
    bais = bais - learning_rate * new_chng_bais




train_prediction = np.zeros((x_train.shape[0] , weight.shape[1]))

for j in range(x_train.shape[0]):
    for k in range(weight.shape[1]):
        total = 0
        for l in range(x_train.shape[1]):
            mul = x_train[j,l] * weight[l,k]
            total = total+mul
        train_prediction[j,k] = total


test_prediction = np.zeros((x_test.shape[0] , weight.shape[1]))

for j in range(x_test.shape[0]):
    for k in range(weight.shape[1]):
        total = 0
        for l in range(x_test.shape[1]):
            mul = x_test[j,l] * weight[l,k]
            total = total+mul
        test_prediction[j,k] = total
test_predicted = bais+test_prediction


train_predicted = bais+train_prediction
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train, train_predicted, color='blue')
plt.title("linear Regression House Vs Price")
plt.xlabel("House Marla")
plt.ylabel("Price")
plt.show()




plt.scatter(x_test,y_test,color='red')
plt.plot(x_test, test_predicted, color='blue')
plt.title("linear Regression Salary Vs Experience")
plt.xlabel("Years of Employee")
plt.ylabel("Saleries of Employee")
plt.show()
