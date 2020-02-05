import numpy as np
import pandas as pd
from numpy import array
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data=pd.read_csv('C:/Users/Snehalreet/Desktop/INTEL-20/TASK-1/Question-1/Salary_Data.csv')
print(data.shape)
X=data['YearsExperience'].values.reshape(-1,1)
Y=data['Salary'].values.reshape(-1,1)
plt.scatter(X,Y,color='red')
model=LinearRegression()
model.fit(X,Y)
plt.plot(X,model.predict(X))
plt.xlabel('Experience(years)')
plt.ylabel('Salary')
plt.show()

