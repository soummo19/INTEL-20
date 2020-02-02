import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading data from dataset
train = pd.read_csv("train.csv")
holdout = pd.read_csv("test.csv")

all_X= train.iloc[:,0].values.reshape(-1,1)
all_y= train.iloc[:,-1].values.reshape(-1,1)

holdout_X = holdout.iloc[:,0].values.reshape(-1,1)
holdout_y = holdout.iloc[:,-1].values.reshape(-1,1)

#Spliting dataset into test and training dataset
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(all_X,all_y,test_size=0.1,random_state=0)

#Training the dataset using linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score




lr = LinearRegression()
lr.fit(train_X,train_y)

scores = cross_val_score(lr,all_X,all_y,cv=5)

prediction = lr.predict(holdout_X)

plt.scatter(holdout_X,prediction,color='green')
plt.plot(holdout_X,holdout_y,color='red')
plt.xlabel("Time Investment")
plt.ylabel("Rs. Earned in Lakhs")
plt.title("Linear Regression Plot")
plt.annotate("R^2 score is {}".format(lr.score()) "\n" "MSE")
plt.savefig("task1reg2.png")








