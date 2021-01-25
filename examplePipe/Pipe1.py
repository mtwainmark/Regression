from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy  as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

us = np.linspace(0, 2 * np.pi, 100)
zs = np.linspace(10, 50, 20)

us, zs = np.meshgrid(us, zs)

xs = 10 * np.cos(us) * np.random.random()
ys = 10 * np.sin(us) * np.random.random()
ax.plot_surface(xs, ys, zs, color='b')

print(xs, "\n", ys,"\n",zs,"\n", us)

plt.show()
