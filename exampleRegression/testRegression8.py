import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Simple method to draw a beautiful cylinder (both radius and height)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Generate cylindrical data with a radius of r on the bottom and a height of h.
# First generate data according to polar coordinates
u = np.linspace(0, 2 * np.pi, 50)  # divide the circle into 50 equal parts
h = np.linspace(0, 1, 20)  # divide the height 1 into 20 parts
x = np.outer(np.sin(u), np.ones(len(h)))  # x value repeated 20 times
y = np.outer(np.cos(u), np.ones(len(h)))  # y value repeated 20 times
z = np.outer(np.ones(len(u)), h)  # x,y corresponding height

# Plot the surface
ax.plot_surface(x, y, z, cmap=plt.get_cmap('rainbow'))

plt.show()