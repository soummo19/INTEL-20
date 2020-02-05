import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data=pd.read_csv('train.csv')
X=data['x'].values.reshape(-1,1)
Y=data['y'].values.reshape(-1,1)

model=LinearRegression()
model.fit(X,Y)

test=pd.read_csv('test.csv')
x=test['x'].values.reshape(-1,1)
y=test['y'].values.reshape(-1,1)

plt.scatter(x,y,color='red')
plt.plot(x,model.predict(x))
plt.xlabel('Time Invested (test input)')
plt.ylabel('money(in lakhs)(Predicted output vs Actual output)')
plt.show()
