# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 22:05:04 2020

@author: hp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading training data set
readdata=pd.read_csv("train.csv")
X= readdata.iloc[:, 0:1].values 
Y= readdata.iloc[:, 1:2].values 

#fitting training dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#reading test dataset
readdata2=pd.read_csv("test.csv")
X_test= readdata2.iloc[:, 0:1].values 
Y_test= readdata2.iloc[:, 1:2].values 

#predicting 
Y_predict=lin_reg.predict(X1)

#plot
plt.scatter(X_test,Y_test,color='blue')
plt.scatter(X_test,Y_predict,color='red')
plt.figtext(.2, .6, "Predicted result = red")
plt.title('time spended vs profit')
plt.xlabel('time spend')
plt.ylabel('profit')
plt.show()
