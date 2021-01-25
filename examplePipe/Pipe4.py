import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib import cm

#Create the profile
Zradii = [0, 1, 5, 10, 12, 14, 16]
ARadii = [1, 1.5, 1, 0.8, 1.3, 0.6, 0.5] #Major axis
BRadii = [0.5, 1, 1.2, 1, 0.5, 0.7, 0.5] # Minor Axis
XOFFSET = [1, 1, 1, 1, 2, 2, 2]    # X Offset of each ellipse
YOFFSET = [1, 1, 1, 1, 2, 2, 2]    # Y Offset of each ellipse

aradius = CubicSpline(Zradii, ARadii, bc_type=((1, 0.5), (1, 0.0)))
bradius = CubicSpline(Zradii, BRadii, bc_type=((1, 0.5), (1, 0.0)))
xoffset = CubicSpline(Zradii, XOFFSET)
yoffset = CubicSpline(Zradii, YOFFSET)

# Make data
npoints=100
thetarange = np.linspace(0, 2 * np.pi, npoints)
zrange = np.linspace(min(Zradii), max(Zradii), npoints)

X = [xoffset(z) + aradius(z)*np.cos(thetarange) for z in zrange]
Y = [yoffset(z) + bradius(z)*np.sin(thetarange) for z in zrange]
Z = np.array([[z] for z in zrange])

# Plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-2, 5)
ax.set_ylim3d(-2, 5)
ax.set_zlim3d(0, 20)
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

#Plot the ellipses
for zz in Zradii:
    XX = xoffset(zz) + aradius(zz)*np.cos(thetarange)
    YY = yoffset(zz) + bradius(zz)*np.sin(thetarange)
    ax.plot(XX,YY,zz, lw=1, color='k')

plt.show();