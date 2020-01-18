import numpy
mydata=numpy.genfromtxt(r'Salary_Data.csv', delimiter=',')
mydata=numpy.delete(mydata,0,axis=0)
mydata=mydata[:,0]
mydata=mydata.T
mydata=mydata.reshape(2,15)
print(numpy.linalg.pinv(mydata))
