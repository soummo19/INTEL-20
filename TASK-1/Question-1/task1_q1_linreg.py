import matplotlib.pyplot as plot
import numpy as npy
from sklearn import lnmd
dat=npy.genfromtxt(r'Salary_Data.csv', delimiter=',')
dat=npy.delete(data,0,axis=0)
x=dat[:,0]
y=dat[:,1]
x=x.reshape(-1,1)
regressn = lnmd.LinearRegression()
regressn.fit(x,y)
predct=regrssn.predict(x)
plot.scatter(x,y,color='orange')
plot.plot(x,predct,color='blue')
plot.xlabel('Years Exp.')
plot.ylabel('Salary')
plot.title('Linear Regression')
plot.show()