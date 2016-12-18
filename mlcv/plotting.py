import cPickle
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import mlcv.input_output as io
import itertools


def plotSVMparam(filename, mode='2d', name='default'):  # filename='../ResultsSVM_poly.pickle'
    results = []
    #file = open(filename, 'r')
    #results = cPickle.load(file)
    results = io.load_object(filename)

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
        print(name + ' 2D: Best parameters are: Degree ' + str(D[ind]) + ' Gamma ' + str(
            G[ind]) + ' with Accuracy ' + str(
            A[ind]))
    elif mode == '2d':
        ax = fig.add_subplot(111)
        ax.plot(results[0], results[2])
        ax.set_xlabel('Gamma')
        ax.set_ylabel('Accuracy')
        G = results[0]
        A = results[2]
        ind = np.argmax(results[2])
        print(name + ' : Best parameters are: Gamma ' + str(G[ind]) + ' with Accuracy ' + str(A[ind]))
    elif mode == 'cost':
        ax = fig.add_subplot(111)
        ax.set_xlabel('Cost')
        ax.set_ylabel('Accuracy')
        ax.plot(results[0], results[1])

        C = results[0]
        A = results[1]
        ind = np.argmax(results[1])
        print(name + '-cost : Best parameters are: C ' + str(C[ind]) + ' with Accuracy ' + str(A[ind]))

    plt.show()


def plotConfusionMatrix(confMatrix, classes, normalize=False):

    plt.figure()
    plt.imshow(confMatrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)


    if normalize:
        confMatrix = confMatrix.astype('float') / confMatrix.sum(axis=1)[:, np.newaxis]
        confMatrix = np.around(confMatrix, 2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(confMatrix)

    thresh = confMatrix.max() / 2.
    for i, j in itertools.product(range(confMatrix.shape[0]), range(confMatrix.shape[1])):
        plt.text(j, i, confMatrix[i, j], horizontalalignment = "center", color = "white" if (confMatrix[i, j] > thresh) else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
