

import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
df = pd.read_csv("Hotel Reservations.csv")

categories = {"Not_Canceled":0, "Canceled":1}
df['booking_status'] = df['booking_status'].replace(categories) 

df = df.drop("Booking_ID" , axis=1) 
df = df.drop("type_of_meal_plan" , axis=1) 
df = df.drop("room_type_reserved" , axis=1) 
df = df.drop("market_segment_type" , axis=1) 




data = df.iloc[: , :-1].values
label = df.iloc[: , 1].values

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(data , label , test_size=0.3 , random_state=0)



data , features = x_train.shape
weight = np.zeros((features,1))
bais = 0   
learning_rate = 0.0015
iteration = 50


for i in range(iteration):
    #sigmoid
    
    z=np.dot(x_train , weight)
    sigmoid_value = 1/(1+np.exp(-z))
    
    #cost function...........
    cost = -(1/data)*np.sum( y_train*np.log(sigmoid_value) + (1-y_train)*np.log(1-sigmoid_value))
    
    #loss = (1/data)(-y_train*np.log(sigmoid_value)-(1-y_train)*np.log(1-sigmoid_value))
    
    a = sigmoid_value.T-y_train.T
    dw = (1/data)*np.dot( a ,x_train)
    db = (1/data)*np.sum(a)
    
    weight = weight.T - learning_rate*dw
    bais = bais - learning_rate*db
    weight = weight.T
    print("iteration ",i , "    cost is ",cost)