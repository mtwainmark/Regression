
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

    ax.scatter3D(x, y, z, c='blue');
    ax.scatter3D(x, y, -(z + 300000), c='blue');
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
        for row in reader:
            print(row)

def gauss(x, y, c1, c2):
    return np.exp(-1 * ((x - c1) ** 2 + (y - c2) ** 2) / 2)


def readCsvWithPandas():
    data = pd.read_csv('example.csv')



if __name__ == '__main__':
    readCsvWithPandas()
