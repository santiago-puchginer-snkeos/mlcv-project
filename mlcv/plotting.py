import cPickle
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plotSVMparam(): #filename='../ResultsSVM_poly.pickle'
    results=[]
    file = open('../ResultsSVM_poly.pickle', 'r')

    results=cPickle.load(file)

    print(results)
    #plt.plot(results[0],results[2])

    #X, Y = np.meshgrid(results[0],results[1])

    fig = plt.figure()

    fig.suptitle('Polynomial', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot_surface(X,Y,results[2], cmap=plt.cm.jet, cstride=1, rstride=1)
    ax.scatter(results[0],results[1], results[2], c='r', marker='o')

    ax.set_xlabel('d')
    ax.set_ylabel('r')
    ax.set_zlabel('Accuracy')

    plt.show()
    D=results[0]
    G=results[1]
    A=results[2]
    ind=np.argmax(results[2])
    print('POLY: Best parameters are: Degree ' +str(D[ind])+ ' Gamma '+str(G[ind])+ ' with Accuracy '+str(A[ind]))

    file = open('../ResultsSVM_rbf_2nd.pickle', 'r')

    results=cPickle.load(file)

    print(results)

    fig = plt.figure()
    fig.suptitle('RBF', fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)

    ax.set_xlabel('Gamma')
    ax.set_ylabel('Accuracy')

    ax.plot(results[0],results[2])

    #X, Y = np.meshgrid(results[0],results[1])
    plt.show()
    D=results[0]
    G=results[0]
    A=results[2]
    ind=np.argmax(results[2])
    #print('Best parameters are: Degree ' +str(D[ind])+ ' Gamma '+str(G[ind])+ ' with Accuracy '+str(A[ind]))
    print('RBF: Best parameters are: Gamma '+str(G[ind])+ ' with Accuracy '+str(A[ind]))

    file = open('../ResultsSVM_sigmoid.pickle', 'r')

    results=cPickle.load(file)

    print(results)
    fig = plt.figure()
    fig.suptitle('Sigmoid', fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)

    ax.set_xlabel('R')
    ax.set_ylabel('Accuracy')
    ax.plot(results[0],results[2])

    #X, Y = np.meshgrid(results[0],results[1])

    plt.show()
    D=results[0]
    G=results[0]
    A=results[2]
    ind=np.argmax(results[2])
    #print('Best parameters are: Degree ' +str(D[ind])+ ' Gamma '+str(G[ind])+ ' with Accuracy '+str(A[ind]))
    print('Sigmoid: Best parameters are: R '+str(G[ind])+ ' with Accuracy '+str(A[ind]))



    ##### Cost sweep
    file = open('../ResultsSVM_poly_cost.pickle', 'r')

    results=cPickle.load(file)

    print(results)
    fig = plt.figure()
    fig.suptitle('Polynomial', fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)

    ax.set_xlabel('Cost')
    ax.set_ylabel('Accuracy')
    ax.plot(results[0],results[1])

    plt.show()
    C=results[0]
    A=results[1]
    ind=np.argmax(results[1])
    print('Poly-cost : Best parameters are: C '+str(C[ind])+ ' with Accuracy '+str(A[ind]))

    file = open('../ResultsSVM_rbf_cost.pickle', 'r')

    results=cPickle.load(file)

    print(results)
    fig = plt.figure()
    fig.suptitle('RBF', fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)

    ax.set_xlabel('Cost')
    ax.set_ylabel('Accuracy')
    ax.plot(results[0],results[1])

    plt.show()
    C=results[0]
    A=results[1]
    ind=np.argmax(results[1])
    print('Rbf-cost : Best parameters are: C '+str(C[ind])+ ' with Accuracy '+str(A[ind]))

    file = open('../ResultsSVM_sigmoid_cost.pickle', 'r')

    results=cPickle.load(file)

    print(results)
    fig = plt.figure()
    fig.suptitle('Sigmoid', fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)

    ax.set_xlabel('Cost')
    ax.set_ylabel('Accuracy')
    ax.plot(results[0],results[1])

    plt.show()
    C=results[0]
    A=results[1]
    ind=np.argmax(results[1])
    print('Sigmoid-cost : Best parameters are: C '+str(C[ind])+ ' with Accuracy '+str(A[ind]))



def plotConfusionMatrix(confMatrix, classes, normalize):

    print 'hello Im still not implemented'


plotSVMparam()