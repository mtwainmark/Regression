import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

a = 3
b = 3
r = 2

stepSize = 0.01

x, y = [], []
X, Y = [], []

t = 0
while t < 2 * math.pi:
    x.append(r * math.cos(t + 0.15) + a)
    y.append(r * math.sin(t) + b)
    t = t + stepSize + (np.random.rand() * 0.2)

fig, ax = plt.subplots()
ax.scatter(x, y)

T = 0
while T < 2 * math.pi:
    X.append(r * math.cos(T) + a)
    Y.append(r * math.sin(T) + b)
    T = T + stepSize

ax.plot(X, Y, color='r')


ax.set(xlabel='X', ylabel='Y', title='Pipe')
ax.grid()
plt.show()

