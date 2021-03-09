import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

x, y, z = [], [], []


def createPipe():
    np.random.seed(16)

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection='3d');

    x = 2 * np.random.random(300) - 1
    y = x + np.random.randn(300) * 200
    z = x ** 2 - y ** 2 + x * y

    ax.scatter3D(x, y, z, c='red');
    ax.scatter3D(x, y, -(z + 300000), c='red');
    plt.show()


def writeCsv():
    i = 0
    fieldnames = ['x', 'y', 'z']

    with open('example.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        dataX = []
        dataY = []
        dataZ = []

        for xs in x:
            dataX.append(xs)
        for ys in y:
            dataY.append(ys)
        for zs in z:
            dataZ.append(zs)

        while i < len(dataX):
            writer.writerow({fieldnames[0]: dataX[i], fieldnames[1]: dataY[i], fieldnames[2]: dataZ[i]})
            i = i + 1

def readCsv():
    with open('example.csv') as File:
        reader = csv.reader(File, delimiter=',', quotechar=',',
                            quoting=csv.QUOTE_MINIMAL)
        mas = []
        for row in reader:
            mas.append(row)
        mas.pop(0)
        print(mas)

def gauss(x, y, c1, c2):
    return np.exp(-1 * ((x - c1) ** 2 + (y - c2) ** 2) / 2)


def readCsvWithPandas():
    data = pd.read_csv('example.csv')

    mas = np.array(data)
    print(mas)
    '''
    x = data['x'].values
    y = data['y'].values

    X, Y = np.meshgrid(x, y)

    Z = np.random.normal(size=X.shape)

    # Plot the data
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm,
                           rstride=1, cstride=1)
    ax.view_init(20, -120)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    
    regr = LinearRegression()

    X = data[['x']].values
    y = data['y'].values
    z = data['z'].values
    quadratic = PolynomialFeatures(degree=2)

    
    cubic = PolynomialFeatures(degree=3)
    X_cubic = cubic.fit_transform(X)
    X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
    X_quad = quadratic.fit_transform(X)

    regr = regr.fit(X_cubic, y)
    y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
    cubic_r2 = r2_score(y, regr.predict(X_cubic))

    regr = regr.fit(X_quad, y)
    y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
    quadratic_r2 = r2_score(y, regr.predict(X_quad))

    plt.scatter(X, y, label='training points', color='lightgray')

    plt.plot(X_fit, y_quad_fit,
             label='quadratic (d=2), $R^2={:.2f}$'.format(quadratic_r2),
             color='red',
             lw=2,
             linestyle='-')

    plt.plot(X_fit, y_cubic_fit,
             label='cubic (d=3), $R^2={:.2f}$'.format(cubic_r2),
             color='green',
             lw=2,
             linestyle='--')

    plt.xlabel('% lower status of the population [LSTAT]')
    plt.ylabel('Price in $1000\'s [MEDV]')
    plt.legend(loc='upper right')
    plt.show()
    '''

if __name__ == '__main__':
    readCsvWithPandas()
