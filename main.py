import numpy as np
from matplotlib import pyplot as plt
from sympy import *
import math

def eval_model(alfa, x):
    ''' вычисление вектора откликов модели
    '''
    return alfa[0] * np.exp(-alfa[1] * (x - alfa[2]) ** 2)

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def jacobian(alfa, x):
    ''' вычисление матрицы Якоби при заданных значениях alfa, x
    '''
    num_points = x.size  # число точек эксперимента
    num_params = alfa.size  # число параметров модели
    J = np.zeros((num_points, num_params))

    for i in range(num_points):
        f = np.exp(-alfa[1] * (x[i] - alfa[2]) ** 2)
        J[i, 0] = f
        J[i, 1] = -(x[i] - alfa[2]) ** 2 * alfa[0] * f
        J[i, 2] = 2 * alfa[1] * (x[i] - alfa[2]) * alfa[0] * f
    return J


def gaussnewton(x, y, alfa, max_iter, stop_norm=1e-10, lam=0):
    ''' оптимизация параметров модели от начального приближения методом Гаусса-Ньютона
    '''

    error_fn = []

    for i in range(max_iter):
        # вычисление вектора откликов
        r = eval_model(alfa, x) - y

        J = jacobian(alfa, x)

        # подстройка параметров модели
        jtj = np.matmul(np.transpose(J), J)

        alfa_new = alfa - np.matmul(
            np.matmul(
                np.linalg.pinv(jtj + lam * np.diag(np.diag(jtj))),
                np.transpose(J)),
            r)

        # норма изменения вектора параметров
        change = np.linalg.norm(alfa_new - alfa) / np.linalg.norm(alfa)

        # норма вектора остатков
        error_fn.append(np.linalg.norm(r))

        print(i, alfa_new, change)

        if change < stop_norm:
            break

        alfa = alfa_new

    if error_fn[-1] < error_fn[0]:
        print('Поиск завершился удачно, ошибка={}'.format(error_fn[-1]))
    else:
        print('Поиск завершился неудачно')

    return alfa, error_fn


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
def calculate_normal_error_point(semi_major, semi_minor, p):
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


# https://all-python.ru/raznoe/proizvodnaya.html
# вычисление частной производной
# для вычисления используем SymPy
def calculate_partial_derivative_A(x, y):
    a, b = symbols('a b')
    return diff(((x ** 2 / a ** 2) + (y ** 2 / b ** 2)), a)


if __name__ == '__main__':
    x0 = 1.5
    y0 = -2
    rx = 7
    ry = 10
    pivot = 0
    points = 50

    # построение точек
    x, y = ellipse_cloud(x0, y0, rx, ry, points, 0.3, pivot)
    # построение эллипса
    xe, ye = ellipse_cloud(x0, y0, rx, ry, points, 0, pivot)

    x_normal_error_out, y_normal_error_out = [], []

    for i in range(len(x)):
        # Вычисление ошибки для каждой точки
        #x_normal_error, y_normal_error = calculate_normal_error_point(x0, y0, [x[i], y[i]])

        x_normal_error_out.append(x[i])
        y_normal_error_out.append(y[i])

    v, b = gaussnewton(np.array(x), np.array(y), np.array([x0, y0, rx, ry]), points)
    print('alfa', v)
    print('error', b)

    xData, yData = ellipse_cloud(v[0], v[1], v[2], v[3], points, 0, pivot)

    plt.scatter(x, y)
    #plt.plot(xe, ye)
    plt.plot(xData, yData)
    plt.axis('equal')
    plt.show()
