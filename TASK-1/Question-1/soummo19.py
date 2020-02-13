import numpy as np
import pandas as pd
from numpy import array
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data=pd.read_csv('D:\Github\OpenCode2020\INTEL-20\TASK-1\Question-1\Salary_Data.csv')
print(data.shape)
X=data['YearsExperience'].values.reshape(-10,10)
Y=data['Salary'].values.reshape(-10,10)
plt.scatter(X,Y,color='green')
model=LinearRegression()
model.fit(X,Y)
plt.plot(X,model.predict(X))
plt.xlabel('Experience(years)')
plt.ylabel('Salary')
plt.show()

