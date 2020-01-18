import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
data=np.genfromtxt(r'C:\Users\HP USER\Desktop\INTEL-20\TASK-1\Question-1\Salary_Data.csv', delimiter=',')
data=np.delete(data,0,axis=0)
x=data[:,0]
y=data[:,1]
x=x.reshape(-1,1)
regr = linear_model.LinearRegression()
regr.fit(x,y)
pred=regr.predict(x)
plt.scatter(x,y,color='red')
plt.plot(x,pred,color='blue')
plt.xlabel('Years Experience')
plt.ylabel('Salary')
plt.title('Linear Regression for Salary Prediction')
plt.show()
