import numpy as np
from matplotlib import pyplot as plt
import math


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


# https://wet-robots.ghost.io/simple-method-for-distance-to-ellipse/
def solve(semi_major, semi_minor, p):
    # координаты точек по модулю
    px = abs(p[0])
    py = abs(p[1])

    # 0.7854
    # atan2(py, px)
    # предполагается, когда p находится внутри эллипса оно возвращает неверные значения
    tx = math.pi / 4
    ty = math.pi / 4

    # центр эллипса
    a = semi_major
    b = semi_minor

    for x in range(0, 3):
        # уравнение эллипса
        x = a * tx
        y = b * ty

        # Эволююта плоской кривой — геометрическое место точек, являющихся центрами кривизны кривой.
        # По отношению к своей эволюте любая кривая является эвольвентой.
        # параметрическая форма
        ex = (a * a - b * b) * tx ** 3 / a
        ey = (b * b - a * a) * ty ** 3 / b

        rx = x - ex
        ry = y - ey

        qx = px - ex
        qy = py - ey

        # возвращает квадратный корень суммы квадратов своих аргументов
        r = math.hypot(ry, rx)
        q = math.hypot(qy, qx)

        # сравниваем и присваеваем минимальное из 1 и из максимального
        tx = min(1, max(0, (qx * r / q + ex) / a))
        ty = min(1, max(0, (qy * r / q + ey) / b))
        # возвращает квадратный корень суммы квадратов своих аргументов
        t = math.hypot(ty, tx)

        tx /= t
        ty /= t
    # copysign - возвращает значение первого аргумента и знак второго аргумента
    return math.copysign(a * tx, p[0]), math.copysign(b * ty, p[1])


if __name__ == '__main__':
    x0 = 1
    y0 = 1
    rx = 12.5
    ry = 14
    pivot = 10

    # построение точек
    x, y = ellipse_cloud(x0, y0, rx, ry, 77, 0.3, pivot)
    # построение эллипса
    xe, ye = ellipse_cloud(x0, y0, rx, ry, 77, 0, pivot)

    x_normal_error_out, y_normal_error_out = [], []

    for i in range(len(x)):
        # Вычисление ошибки для каждой точки
        x_normal_error, y_normal_error = solve(x0, y0, [x[i], y[i]])
        x_normal_error_out.append(x[i] + x_normal_error)
        y_normal_error_out.append(y[i] + y_normal_error)

    plt.plot(x_normal_error_out, y_normal_error_out, 'r')

    plt.scatter(x, y)
    plt.plot(xe, ye, 'y')
    plt.axis('equal')
    plt.show()
