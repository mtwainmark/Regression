import numpy as np
from matplotlib import pyplot as plt


def ellipse_cloud(x0, y0, rx, ry, npoints, noise, pivot, regular=True):
    if regular:
        t = np.linspace(0, 2 * np.pi, npoints)
    else:
        t = np.random.uniform(0, 2 * np.pi, npoints)

    x = rx * np.cos(t) + x0
    y = ry * np.sin(t) + y0

    # погрешность вносит смещение координат вдоль прямой, соединяющей центр и точку на окружности
    pog = np.random.normal(loc=0, scale=noise, size=npoints)
    h = np.hypot(x, y)
    pogx = pog * x / h
    pogy = pog * y / h

    # поворот оси
    x_pivot = x * np.cos(pivot) + y * np.sin(pivot)
    y_pivot = -x * np.sin(pivot) + y * np.cos(pivot)

    return x_pivot + pogx, y_pivot + pogy


if __name__ == '__main__':
    x0 = 1
    y0 = 1
    rx = 12.5
    ry = 7
    pivot = 0.5

    x, y = ellipse_cloud(x0, y0, rx, ry, 77, 0.3, pivot)
    xe, ye = ellipse_cloud(x0, y0, rx, ry, 500, 0, pivot)

    plt.scatter(x, y)
    plt.plot(xe, ye, 'y')
    plt.axis('equal')
    plt.show()
