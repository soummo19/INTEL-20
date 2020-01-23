import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


data=np.genfromtxt(r'Salary_Data.csv', delimiter=',')
data=np.delete(data,0,axis=0)
x=data[:,0]
y=data[:,1]
x=x.reshape(-1,1)
reg = LinearRegression()
reg.fit(x, y)
plt.scatter(x, y, color = 'red')
plt.plot(x, reg.predict(x), color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
