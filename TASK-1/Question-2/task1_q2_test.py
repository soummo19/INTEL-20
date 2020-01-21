
import numpy as npy
import matplotlib.pyplot as plot 
import pandas as pnds


x_train = pnds.read_csv("train.csv")
x_test = pnds.read_csv("test.csv")
x_trset = x_train.iloc[:, :-1].values
y_trset = x_train.iloc[:, -1].values
x_teset = x_test.iloc[:, :-1].values
y_teset = x_test.iloc[:, -1].values


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_trset, y_trset)

#predicted values for the test 
y_pred = regressor.predict(x_teset)

# Plotting and
plot.scatter(x_teset, y_teset, color = 'orange')
plot.scatter(x_teset, y_pred, color = 'blue')
plot.figtext(.2, .6, "Predicted result = blue")
plot.figtext(.2, .7, "Test set result = orange")
plot.xlabel('Time invested')
plot.ylabel('Money(lk)')
plt.show 