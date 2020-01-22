from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

dataset = np.genfromtxt('Salary_Data.csv', delimiter=',')
dataset = np.delete(dataset, 0, axis=0)
x = dataset[:,0].reshape(-1,1)
y = dataset[:,1]

regr = LinearRegression()
regr.fit(x,y)

plt.scatter(x, y, color='red')
plt.plot(x, regr.predict(x), color='blue')
plt.xlabel('Experience(Years)')
plt.ylabel('Salary')
plt.title('Linear Regression(Task-1 Q1)')
plt.show()