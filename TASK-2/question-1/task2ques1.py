
import numpy as np
data = np.genfromtxt('../../Task-1/Question-1/Salary_Data.csv', delimiter=',')
data = data[:,0][1:26]
matrix = data.reshape(5,5)
inv_matrix = np.linalg.inv(matrix)
print(inv_matrix)
