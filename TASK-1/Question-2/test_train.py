# Importing libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

# Importing the dataset
x_train = pd.read_csv("train.csv")
x_test = pd.read_csv("test.csv")
x_trset = x_train.iloc[:, :-1].values
y_trset = x_train.iloc[:, -1].values
x_teset = x_test.iloc[:, :-1].values
y_teset = x_test.iloc[:, -1].values

# Training the dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_trset, y_trset)

# Getting the predicted values for the test set
y_pred = regressor.predict(x_teset)

# Plotting and compairing the results
plt.scatter(x_teset, y_teset, color = 'red')
plt.scatter(x_teset, y_pred, color = 'blue')
plt.figtext(.2, .6, "Predicted result = blue")
plt.figtext(.2, .7, "Test set result = red")
plt.xlabel('Time invested')
plt.ylabel('Money(in lakhs)')
plt.show