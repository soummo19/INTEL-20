import numpy as n
import pandas as p
import matplotlib.pyplot as plt

#reading training data set
data=p.read_csv('train.csv')
X_train= data.iloc[:, :-1].values 
Y_train= data.iloc[:, 1].values 

#fitting training dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,Y_train)

#reading test dataset
data1=p.read_csv("test.csv")
X_test= data1.iloc[:, :-1].values 
Y_test= data1.iloc[:, 1].values 

#predicting 
Y_predict=lin_reg.predict(X_test)

#plot
plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_train,lin_reg.predict(X_train),color='orange')
plt.figtext(.2, .7, "Test set result = purple")
plt.figtext(.2, .6, "Predicted result = orange")
plt.title('Linear Regression Model')
plt.ylabel('Amount (in lakhs)')
plt.xlabel('Time invested')
plt.show()
