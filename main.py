import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from sympy import *
import math
import scipy


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


def calculate_distance(x0, y0, rx, ry, point):
    dx = point[0] - x0
    dy = point[1] - y0

    k = dy / dx
    w = (1 / rx ** 2) + (k ** 2 / ry ** 2)
    w = 1 / w

    a = 1
    b = -2 * x0
    c = x0 ** 2 - w

    d = b ** 2 - 4 * a * c

    d = math.sqrt(d)

    x = (-b + d) / (2 * a)
    x1 = (-b - d) / (2 * a)

    y = (x - x0) * (dy / dx) + y0
    y1 = (x1 - x0) * (dy / dx) + y0

    a = abs(point[0] - x)
    b = abs(point[0] - x1)

    if a < b:
        return [x, y]
    else:
        return [x1, y1]


def calculate_error(point, point_ellipse):
    return math.sqrt((point[0] - point_ellipse[0]) ** 2 + (point[1] - point_ellipse[1]) ** 2)

def view_3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, label='points')
    ax.scatter(x_normal_error_out, y_normal_error_out, color='g', label='distance')
    ax.scatter(xe, ye, label='ideal_ellipse')
    # ax.axis('equal')
    ax.legend()

if __name__ == '__main__':
    x0 = 1.5
    y0 = -2
    rx = 7
    ry = 10
    pivot = 0
    points = 50

    # построение точек
    x, y = ellipse_cloud(x0, y0, rx, ry, points, 0.2, pivot)
    # построение эллипса
    xe, ye = ellipse_cloud(x0, y0, rx, ry, points, 0, pivot)

    x_normal_error_out, y_normal_error_out = [], []

    for i in range(len(x)):
        # Вычисление ошибки для каждой точки с помощью центров эллипса
        #x_normal_error, y_normal_error = calculate_normal_error_point(x0, y0, [x[i], y[i]])
        # Вычисление ошибки с помощью канонического уравнения эллипса
        normal_point_x, normal_point_y = calculate_distance(x0, y0, rx, ry, [x[i], y[i]])

        #popt, pcov = curve_fit(func, normal_point_x, normal_point_y)
        #print('popt', popt)

        # Расчет расстояния от точки до эллипса
        #print(calculate_error([x[i], y[i]], calculate_distance(x0, y0, rx, ry, [x[i], y[i]])))

        x_normal_error_out.append(normal_point_x)
        y_normal_error_out.append(normal_point_y)

    popt, pcov = gaussnewton(np.array(x_normal_error_out), np.array(y_normal_error_out), np.array([x0, y0, rx, ry]), 5)

    z, z1 = ellipse_cloud(popt[0], popt[1], popt[2], popt[3], points, 0, pivot)
    #plt.scatter(z, z1, color='r', label='gauss-newton')

    plt.scatter(x, y,  label='points')
    plt.scatter(x_normal_error_out, y_normal_error_out, color='g', label='distance')
    plt.scatter(xe, ye, label='ideal_ellipse')
    plt.legend()
    view_3d()
    plt.show()
