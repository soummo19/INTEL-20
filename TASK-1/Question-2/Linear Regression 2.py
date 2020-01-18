import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics,linear_model
trainingdata=np.genfromtxt(r'train.csv',delimiter=',')
trainingdata=np.delete(trainingdata,0,axis=0)
trainX=trainingdata[:,0]
trainY=trainingdata[:,1]
testdata=np.genfromtxt(r'test.csv',delimiter=',')
testdata=np.delete(testdata,0,axis=0)
testX=testdata[:,0]
testY=testdata[:,1]
trainX=trainX.reshape(-1,1)
testX=testX.reshape(-1,1)
regr=linear_model.LinearRegression()
regr.fit(trainX,trainY)
testYpred=regr.predict(testX)
trainYpred=regr.predict(trainX)
plt.scatter(trainX,trainY,color='red')
plt.plot(trainX,trainYpred,color='blue')
plt.xlabel('Time Invested')
plt.ylabel('Amount(in lakhs)')
plt.title('Linear Regression on training data')
print('R2 Score On Test Data = ',metrics.r2_score(testY,testYpred))
plt.show()




