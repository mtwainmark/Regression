import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from sympy import *
import math
import pandas as pd

def eval_model(alfa, x):
    ''' вычисление вектора откликов модели
    '''
    return alfa[0] * np.exp(-alfa[1] * (x - alfa[2]) ** 2)


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

def jacobianMY(alfa, x):
    ''' вычисление матрицы Якоби при заданных значениях alfa, x
    '''
    num_points = x.size  # число точек эксперимента
    num_params = alfa.size  # число параметров модели
    J = np.zeros((num_points, num_params))

    for i in range(num_points):
        f = ((((x - alfa[0]) + (y - alfa[1])) / alfa[2]) ** 2
             + (((x - alfa[0]) - (y - alfa[1])) / alfa[3]) ** 2)
        J[i, 0] = (-2*x + 2*alfa[0] + 2*y - 2*alfa[1])/alfa[3]**2 + (-2*x + 2*alfa[0] - 2*y + 2*alfa[1])/alfa[2]**2
        J[i, 1] = (2*x - 2*alfa[0] - 2*y + 2*alfa[1])/alfa[3]**2 + (-2*x + 2*alfa[0] - 2*y + 2*alfa[1])/alfa[2]**2
        J[i, 2] = -2*(x - alfa[0] + y - alfa[1])**2/alfa[2]**3
        J[i, 3] = -2*(x - alfa[0] - y + alfa[1])**2/alfa[3]**3
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

        #print(i, alfa_new, change)

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
def calculate_partial_derivative():
    x, y, a, b, x0, y0 = symbols('x, y, a, b, x0, y0')
    f = ((((x - x0) + (y - y0)) / a) ** 2
         + (((x - x0) - (y - y0)) / b) ** 2)
    return print(diff(f, b))

# https://ip76.ru/theory-and-practice/inellipse-line/
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

    #ax.scatter(x, y, label='points')
    #ax.scatter(x_normal_error_out, y_normal_error_out, color='g', label='distance')
    #ax.scatter(xe, ye, label='ideal_ellipse')
    #ax.plot(curve_fit_x, curve_fit_y, color='pink', label='gauss-newton2')
    cols = ['x', 'y', 'z', 'a', 'b', 'c']
    points1 = pd.read_csv('points/cloud1.xyz', sep=' ', header=None, names=cols)
    points2 = pd.read_csv('points/cloud2.xyz', sep=' ', header=None, names=cols)
    points3 = pd.read_csv('points/cloud3.xyz', sep=' ', header=None, names=cols)
    points4 = pd.read_csv('points/cloud4.xyz', sep=' ', header=None, names=cols)
    points5 = pd.read_csv('points/cloud5.xyz', sep=' ', header=None, names=cols)
    points6 = pd.read_csv('points/cloud6.xyz', sep=' ', header=None, names=cols)

    df = pd.concat([points1, points2, points3, points4, points5, points6], ignore_index=True, sort=False)
    df = df.sort_values(by=['y'])

    n = np.sort(list(set(df['y'].values)))

    #mask = (b['y'] <= 2.85047629) & (b['y'] > 2.65030425)

    mm, mm1 = [], []

    '''
    for k in range(255, 260):
        mask = df['y'] == n[k]
        ax.scatter(df['x'][mask].values, df['y'][mask].values, df['z'][mask].values)

        pp, pcov1 = curve_fit(func, (df['x'][mask].values, df['z'][mask].values), np.ones(len(df['x'][mask].values)),
                          maxfev=10000000, method='lm')

        normal_point_x1, normal_point_y1 = calculate_distance(pp[2], pp[3], pp[0], pp[1], [df['x'][k], df['z'][k]])
        mm.append(normal_point_x1)
        mm1.append(normal_point_y1)
        ax.scatter(normal_point_x1, df['y'][k], normal_point_y1)
    '''



    #mask = b['y'] <= 2.75
    mask = (df['y'] >= 4.7) & (df['y'] <= 4.7009)

    ax.scatter(df['x'][mask].values, df['y'][mask].values, df['z'][mask].values)

    pp, pcov1 = curve_fit(func, (df['x'][mask].values, df['z'][mask].values), np.ones(len(df['x'][mask].values)), maxfev=10000000, method='lm')

    curve_fit_x, curve_fit_y = ellipse_cloud(pp[2], pp[3], pp[0], pp[1], 200, 0, 0)

    #сравнение полуосей эллипса
    print(int(pp[0]) == int(pp[1]))

    ax.scatter(curve_fit_x, 4.7007283, curve_fit_y)

    for i in range(len(df['x'][mask].values)):
        normal_point_x1, normal_point_y1 = calculate_distance(pp[2], pp[3], pp[0], pp[1], [df['x'][i], df['z'][i]])
        error = calculate_error([df['x'][i], df['z'][i]], [normal_point_x1, normal_point_y1])

        if error > 0.8:
            mm.append(normal_point_x1)
            mm1.append(normal_point_y1)

    ax.scatter(mm, 4.7007283, mm1)


# https://stackoverflow.com/questions/59042588/multivariate-curve-fitting-in-python-for-estimating-the-parameter-and-order-of-e
def func(data, a, b, h, k, A):
    x, y = data
    return ((((x - h) * np.cos(A) + (y - k) * np.sin(A)) / a) ** 2
            + (((x - h) * np.sin(A) - (y - k) * np.cos(A)) / b) ** 2)

def func1(data, a, b, h, k, A):
    x, y = data
    return ((((x - h) * np.cos(A) + (y - k) * np.sin(A)) / a) ** 2
            + (((x - h) * np.sin(A) - (y - k) * np.cos(A)) / b) ** 2)

def read():
    cols = ['x', 'y', 'z', 'a', 'b', 'c']
    points = pd.read_csv('points/cloud1.xyz', sep=' ', header=None, names=cols)


if __name__ == '__main__':
    x0 = 1.5
    y0 = -2
    rx = 7
    ry = 10
    pivot = 0
    points = 50

    #скользящее среднее по заданному количеству точек + пороговая функция
    #сварной вмятины эллиптичность

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

    #popt, pcov = gaussnewton(np.array(x_normal_error_out), np.array(y_normal_error_out), np.array([[x_normal_error_out, y_normal_error_out], rx, ry, x0, y0, 1]), 5)

    #z, z1 = ellipse_cloud(popt[0], popt[1], popt[2], popt[3], points, 0, pivot)
    #plt.scatter(z, z1, color='r', label='gauss-newton')

    #pp, pcov1 = curve_fit(func, (x_normal_error_out, y_normal_error_out), np.ones(points), method='lm')

    #print('pp', pcov1)

    #curve_fit_x, curve_fit_y = ellipse_cloud(pp[2], pp[3], pp[0], pp[1], points, 0, pivot)

    #plt.scatter(x, y,  label='points')
    #plt.scatter(x_normal_error_out, y_normal_error_out, color='g', label='distance')
    #plt.scatter(xe, ye, label='ideal_ellipse')
    #plt.legend()
    #plt.plot(curve_fit_x, curve_fit_y, color='pink', label='gauss-newton2')
    view_3d()
    plt.show()
