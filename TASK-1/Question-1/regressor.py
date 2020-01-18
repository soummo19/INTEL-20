# Importing the libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Fit regressor model in dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)

# Plotting the trained model
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show