import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Getting data from file
data=np.genfromtxt(r'Salary_Data.csv', delimiter=',')
data=np.delete(data,0,axis=0)
# print (data)
X=data[:,0]
# print(np.size(X))
y=data[:,1]
X=X.reshape(-1,1)
# print(X)

#appplying linear Regression
linReg=LinearRegression()
linReg.fit(X,y)
pre=linReg.predict(X)

#plotting data
plt.scatter(X,y,c="red")
plt.plot(X,pre,scalex=True,scaley=True, color='blue')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Model for Salary")
plt.show()