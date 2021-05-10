import numpy as np
from numpy.random import default_rng
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def func(data, a, b, h, k, A):
    x, y = data
    return ((((x - h) * np.cos(A) + (y - k) * np.sin(A)) / a) ** 2
            + (((x - h) * np.sin(A) - (y - k) * np.cos(A)) / b) ** 2)


rng = default_rng(3)
numPoints = 50
center = rng.random(2) * 10 - 5
theta = rng.uniform(0, 2 * np.pi, (numPoints, 1))
circle = np.hstack([np.cos(theta), np.sin(theta)])
ellipse = (circle.dot(rng.random((2, 2)) * 2 * np.pi - np.pi)
           + (center[0], center[1]) + rng.normal(0, 1, (50, 2)) / 1e1)
pp, pcov = curve_fit(func, (ellipse[:, 0], ellipse[:, 1]), np.ones(numPoints),
                     p0=(1, 1, center[0], center[1], np.pi / 2),
                     method='dogbox')
plt.scatter(ellipse[:, 0], ellipse[:, 1], label='Data Points')
plt.gca().add_patch(Ellipse(xy=(pp[2], pp[3]), width=2 * pp[0],
                            height=2 * pp[1], angle=pp[4] * 180 / np.pi,
                            fill=False))
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()