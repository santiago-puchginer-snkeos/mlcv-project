from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras import backend as K
from keras.utils.visualize_util import plot

import numpy as np
import matplotlib.pyplot as plt
import cPickle
import os
import time
import argparse
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import itertools

import mlcv.input_output  as io
import mlcv.kernels as kernels
from libraries.yael.yael import ynumpy
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import sklearn.decomposition as decomposition
import sklearn.preprocessing as preprocessing

""" PARAMETER SWEEP """

params_distribution = {
    'C': np.logspace(-5, 3, 10 ** 6)
}
n_iter = 100


def train():
    best_accuracy = 0
    best_params = {}
    cv_results = {}

    base_model = VGG16(weights='imagenet')

    # crop the model up to a certain layer
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)

    # Read the training set
    train_images_filenames = cPickle.load(open('./dataset/train_images_filenames.dat', 'r'))
    test_images_filenames = cPickle.load(open('./dataset/test_images_filenames.dat', 'r'))
    train_labels = cPickle.load(open('./dataset/train_labels.dat', 'r'))
    test_labels = cPickle.load(open('./dataset/test_labels.dat', 'r'))
    io.log('\nLoaded {} train images.'.format(len(train_images_filenames)))
    io.log('\nLoaded {} test images.'.format(len(test_images_filenames)))

    # read and process training images
    print 'Getting features from training images'
    start_feature = time.time()

    first = 1
    for i in range(len(train_images_filenames)):
        img = image.load_img(train_images_filenames[i], target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # get the features from images
        features = model.predict(x)
        features = features[0, :]
        if first == 1:
            Desc = features
            first = 0
        else:
            Desc = np.vstack((Desc, features))

    feature_time = time.time() - start_feature
    io.log('Elapsed time: {:.2f} s'.format(feature_time))

    io.log('Scaling features...')
    start_scaler = time.time()
    stdSlr = StandardScaler().fit(Desc)
    D_scaled = stdSlr.transform(Desc)
    scaler_time = time.time() - start_scaler
    io.log('Elapsed time: {:.2f} s'.format(scaler_time))

    io.log('Optimizing SVM hyperparameters...')
    start_crossvalidation = time.time()
    svm = SVC(kernel='precomputed', probability=True)
    random_search = RandomizedSearchCV(
        svm,
        params_distribution,
        n_iter=n_iter,
        scoring='accuracy',
        refit=False,
        cv=3,
        verbose=1
    )

    # Precompute Gram matrix
    gram = kernels.intersection_kernel(D_scaled, D_scaled)
    random_search.fit(gram, train_labels)
    crossvalidation_time = time.time() - start_crossvalidation
    io.log('Elapsed time: {:.2f} s'.format(crossvalidation_time))

    # Convert MaskedArrays to ndarrays to avoid unpickling bugs
    results = random_search.cv_results_
    results['param_C'] = results['param_C'].data

    # Appending all parameter-scores combinations
    cv_results.update({
            'cv_results': results,
            'feature_time': feature_time,
            'scaler_time': scaler_time,
            'crossvalidation_time': crossvalidation_time,
            'total_time': feature_time + scaler_time + crossvalidation_time
    })
    io.save_object(cv_results, 'intersection_svm_CNNfeatures', ignore=True)
    print('Best accuracy ' + str(random_search.best_score_))
    # Obtaining the parameters which yielded the best accuracy
    if random_search.best_score_ > best_accuracy:
        best_accuracy = random_search.best_score_
        best_params = random_search.best_params_


    io.log('-------------------------------\n')
    io.log('\nSaving best parameters...')
    io.save_object(best_params, 'best_params_intersection_svm_CNNfeatures', ignore=True)
    best_params_file = os.path.abspath('./ignore/best_params_intersection_svm_CNNfeatures.pickle')
    io.log('Saved at {}'.format(best_params_file))

    io.log('\nSaving all cross-validation values...')
    io.save_object(cv_results, 'intersection_svm_CNNfeatures', ignore=True)
    cv_results_file = os.path.abspath('./ignore/intersection_svm_CNNfeatures.pickle')
    io.log('Saved at {}'.format(cv_results_file))

    io.log('\nBEST PARAMS')
    io.log('C={} --> accuracy: {:.3f}'.format(
        best_params['C'],
        best_accuracy
    ))

def plot_curve():
    io.log('Loading cross-validation values...')
    cv_values = io.load_object('intersection_svm_CNNfeatures', ignore=True)

    io.log('Loading best parameters...')
    best_params = io.load_object('best_params_intersection_svm_CNNfeatures', ignore=True)

    io.log('Plotting...')
    colors = itertools.cycle(
        ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'darkolivegreen', 'darkviolet', 'black']
    )


    # Subplot parameters
    plt.figure(facecolor='white')
    num_columns = 1

    # All subplots
    # Search dictionary
    val = cv_values
    results = val['cv_results']
    feature_time = val['feature_time']
    scaler_time = val['scaler_time']
    crossvalidation_time = val['crossvalidation_time']
    total_time = val['total_time']

    # Plot
    x = results['param_C']
    y = results['mean_test_score']
    e = results['std_test_score']
    sorted_indices = x.argsort()
    x_sorted = np.asarray(x[sorted_indices], dtype=np.float64)
    y_sorted = np.asarray(y[sorted_indices], dtype=np.float64)
    e_sorted = np.asarray(e[sorted_indices], dtype=np.float64)
    color = colors.next()
    ax = plt.subplot(1, 1, 1)
    ax.set_xscale("log")
    ax.set_ylim((0, 1))
    ax.errorbar(x_sorted, y_sorted, e_sorted, linestyle='--', lw=2, marker='x', color=color)
    ax.set_xlabel('C')
    ax.set_ylabel('Accuracy')

    # Print information
    io.log('-------------')
    io.log('Mean accuracy: {}'.format(y.max()))
    io.log('Std accuracy: {}'.format(e[np.argmax(y)]))
    io.log('C: {}'.format(x[np.argmax(y)]))
    io.log()
    io.log('Timing')
    io.log('\tSIFT time: {:.2f} s'.format(feature_time))
    io.log('\tScaler time: {:.2f} s'.format(scaler_time))
    io.log('\tCV time: {:.2f} s'.format(crossvalidation_time))
    io.log('\t_________________________')
    io.log('\tTOTAL TIME: {:.2f} s'.format(total_time))
    io.log()
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--type', default='plot', choices=['train', 'plot'])
    args = args_parser.parse_args()
    exec_option = args.type

    if exec_option == 'train':
        train()
    elif exec_option == 'plot':
        plot_curve()
    exit(0)