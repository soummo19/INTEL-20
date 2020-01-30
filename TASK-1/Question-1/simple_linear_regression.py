import numpy as n
import pandas as p
import matplotlib.pyplot as plt

#reading data set
data=p.read_csv('Salary_Data.csv')
X= data.iloc[:, :-1].values 
Y= data.iloc[:, 1].values 

#split the data set for training the model
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train,Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)

#fitting training dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,Y_train)

#predicting 
Y_predict=lin_reg.predict(X_test)

#plot
plt.scatter(X_train,Y_train,color='purple')
plt.plot(X_train,lin_reg.predict(X_train),color='green')
plt.title('Salary vs Experience (Linear Regression)')
plt.xlabel('Year of experience')
plt.ylabel('Salary')
plt.show()
