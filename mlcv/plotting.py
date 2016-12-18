import itertools

import matplotlib.pyplot as plt
import numpy as np

import mlcv.input_output as io


def plot_svm_param(filename, mode='2d', name='default'):
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


def plot_confusion_matrix(conf_matrix, classes, normalize=False):
    plt.figure()
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        conf_matrix = np.around(conf_matrix, 2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(conf_matrix)

    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, conf_matrix[i, j], horizontalalignment="center",
                 color="white" if (conf_matrix[i, j] > thresh) else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
