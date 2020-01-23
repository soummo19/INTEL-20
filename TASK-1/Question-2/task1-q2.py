import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn import linear_model


trdata=np.genfromtxt(r'train.csv',delimiter=',')
tedata=np.genfromtxt(r'test.csv',delimiter=',')

trdata=np.delete(trdata,0,axis=0)
tedata=np.delete(tedata,0,axis=0)

trX=trdata[:,0]
trY=trdata[:,1]

teX=tedata[:,0]
teY=tedata[:,1]

trX=trX.reshape(-1,1)
teX=teX.reshape(-1,1)

reg=linear_model.LinearRegression()
reg.fit(trX,trY)

plt.scatter(trX,trY,color='orange')
plt.plot(trX,reg.predict(trX),color='blue')
plt.xlabel('Time')
plt.ylabel('Money (lac)')
plt.title('Linear-Regression')

print('Score On Test Data = ',metrics.r2_score(teY,reg.predict(teX)))

plt.show()
