import numpy as npy 
dat=npy.genfromtxt(r'Salary_data.csv',delimeter=',')#reads data from Salary_data.csv separated by ','
dat=npy.delete(dat,0,axis=0)
dat=dat[:,0]
dat=dat.T #takes transpose
dat=dat.reshape(2,15)
print(npy.linalg.pinv(dat))
