import numpy as np

dataset = np.genfromtxt('../../TASK-1/Question-1/Salary_Data.csv', delimiter=',')
dataset = np.delete(dataset,0,axis=0)
dataset = dataset[:,0]
dataset = dataset.T
dataset = dataset.reshape(2,15)

print(np.linalg.pinv(dataset))
