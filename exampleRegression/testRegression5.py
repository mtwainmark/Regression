import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# your model definition
def model(z, a, b):
    return a * np.exp(-b * z)

# your input data
x = np.array([20, 30, 40, 50, 60])
y = np.array([5.4, 4.0, 3.0, 2.2, 1.6])

# do the fit with some initial values
popt, pcov = curve_fit(model, x, y, p0=(5, 0.1))

# prepare some data for a plot
xx = np.linspace(20, 60, 1000)
yy = model(xx, *popt)

plt.plot(x, y, 'o', xx, yy)
plt.title('Exponential Fit')

plt.show()