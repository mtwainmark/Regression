import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import pandas as pd
import csv

data = pd.read_csv('E:\Study\Regression\example.csv')

mas = np.array(data)
print(mas)
coefficient = mas[:,0:2]
dependent = mas[:,-1]

def model(p,x):
    a,b,c = p
    u = x[:,0]
    v = x[:,1]
    return (a*u**2 + b*v + c)

def residuals(p, y, x):
    a,b,c = p
    err = y - model(p,x)
    return err

p0 = np.array([2,3,4]) #some initial guess

p = leastsq(residuals, p0, args=(dependent, coefficient))[0]

def f(p,x):
    return p[0]*x[0] + p[1]*x[1] + p[2]

i = []

for x in coefficient:
    i.append(f(p,x))

ax = plt.axes(projection='3d')
ax.scatter3D(mas[:,0], mas[:,1], mas[:,2], c='red')
ax.scatter3D(mas[:,0], mas[:,1], -(mas[:,2]+300000), c='red');

#plt.plot(mas[:,0], mas[:,1], i[:],
         color='blue',
         lw=2,
         linestyle='--')


plt.show()
