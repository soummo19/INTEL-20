# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 02:45:13 2020

@author: hp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#reading data from dataset
readdata=pd.read_csv("Salary_Data.csv")
X= readdata.iloc[: , :-1].values
Y= readdata.iloc[: , 1].values

#spliting dataset into test and training dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

#training the dataset using linear regression model
from sklearn.linear_model import LinearRegression
reg= LinearRegression()
reg.fit(X_train,Y_train)

#predicting the output
Y_pred=reg.predict(X_test)

#ploting graph
plt.scatter(X,Y,color='blue')
plt.plot(X,reg.predict(X),color='red')
plt.title('predicting salary using Linear Regression')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()
