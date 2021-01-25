import numpy as np
import matplotlib.pyplot as plt
import csv


np.random.seed(16)

fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection='3d');

x = 2*np.random.random(300) - 1
y = x + np.random.randn(300) * 200
z = x**2 - y**2 + x*y

ax.scatter3D(x, y, z, c='blue');
ax.scatter3D(x, y, -(z + 300000), c='blue');


plt.show()
