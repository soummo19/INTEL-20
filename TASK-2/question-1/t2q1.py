import numpy as np
data=np.genfromtxt('../../Task-1/Question-1/Salary_Data.csv', delimiter=',')
data=np.delete(data,0,axis=0)
data=data[:,0]
data=data.T
data=data.reshape(2,15)
print(np.linalg.pinv(data))
