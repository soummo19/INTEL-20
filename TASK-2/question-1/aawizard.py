import numpy as np

data=np.genfromtxt('../../Task-1/Question-1/Salary_Data.csv', delimiter=',')
data=np.delete(data,0,axis=0)
#transpose of first column
data=data[0:25,0]
data=data.T
print("Transpose of first column")
print(data.T)
# print(data.size)
# data=data[0:25,:]

#reshaping the matrix
data=data.reshape(5,5)
print("\nReshaped array into matrix")
print(data)

#printing the inverse
print(np.linalg.pinv(data))