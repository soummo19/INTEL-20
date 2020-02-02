import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#reading data from dataset
readdata=pd.read_csv("Salary_Data.csv")
X= readdata.iloc[: , :-1].values
Y= readdata.iloc[: , 1].values

#Spliting dataset into test and training dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

#Training the dataset using linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

lr= LinearRegression()
lr.fit(X_train,Y_train)

scores = cross_val_score(lr,X,Y,cv=10)
accuracy = scores.mean()

#ploting graph
plt.scatter(X,Y,color='green')
plt.plot(X,reg.predict(X),color='red')
plt.title('Linear Regression Model')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
plt.savefig('task1regplot')



