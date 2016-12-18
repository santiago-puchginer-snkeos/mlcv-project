import cPickle
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import mlcv.input_output as io


def plotSVMparam(filename, mode='2d', name='default'):  # filename='../ResultsSVM_poly.pickle'
    results = []
    file = open(filename, 'r')

    results = cPickle.load(file)

    print(results)
    fig = plt.figure()
    fig.suptitle(name, fontsize=14, fontweight='bold')
    if mode == '3d':
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(results[0], results[1], results[2], c='r', marker='o')
        if name == 'poly':
            ax.set_xlabel('d')
            ax.set_ylabel('r')
            ax.set_zlabel('Accuracy')

        elif name == 'sigmoid':

            ax.set_xlabel('r')
            ax.set_ylabel('gamma')
            ax.set_zlabel('Accuracy')

        D = results[0]
        G = results[1]
        A = results[2]
        ind = np.argmax(results[2])
        print(name+' 2D: Best parameters are: Degree ' + str(D[ind]) + ' Gamma ' + str(
            G[ind]) + ' with Accuracy ' + str(
            A[ind]))
    elif mode=='2d':
        ax = fig.add_subplot(111)
        ax.plot(results[0], results[2])
        ax.set_xlabel('Gamma')
        ax.set_ylabel('Accuracy')
        G = results[0]
        A = results[2]
        ind = np.argmax(results[2])
        print(name+ ' : Best parameters are: Gamma ' + str(G[ind]) + ' with Accuracy ' + str(A[ind]))
    elif mode=='cost':
        ax = fig.add_subplot(111)
        ax.set_xlabel('Cost')
        ax.set_ylabel('Accuracy')
        ax.plot(results[0], results[1])

        C = results[0]
        A = results[1]
        ind = np.argmax(results[1])
        print(name+'-cost : Best parameters are: C ' + str(C[ind]) + ' with Accuracy ' + str(A[ind]))

    plt.show()


def plotConfusionMatrix(confMatrix, classes, normalize):
    print 'hello Im still not implemented'

