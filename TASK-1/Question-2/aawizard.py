import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Getting data from file
data=np.genfromtxt(r'train.csv', delimiter=',')
data=np.delete(data,0,axis=0)
dataTest=np.genfromtxt(r'test.csv', delimiter=',')
dataTest=np.delete(dataTest,0,axis=0)
# print (data)
Xtrain=data[:,0]
Xtest=dataTest[:,0]
# print(np.size(Xtrain))
# print(np.size(Xtest))
ytrain=data[:,1]
ytest=dataTest[:,1]
Xtrain=Xtrain.reshape(-1,1)
Xtest=Xtest.reshape(-1,1)
# print(np.size(Xtrain))
# print(np.size(Xtest))

#appplying linear Regression
linReg=LinearRegression()
linReg.fit(Xtrain,ytrain)
trainYpre=linReg.predict(Xtrain)

#testing on test set
testYpre=linReg.predict(Xtest)

#plotting data
plt.scatter(Xtrain, ytrain, color = 'red')
plt.scatter(Xtest, testYpre, color = 'blue')
plt.figtext(.2, .6, "Predicted result = red")
plt.figtext(.2, .7, "Test set result = blue")
plt.xlabel("Time Invested")
plt.ylabel("Amount(in lakhs)")
plt.title("Linear Regression Model")
plt.show()