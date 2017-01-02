from __future__ import print_function, division

import time
import argparse
import itertools


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import mlcv.bovw as bovw
import mlcv.feature_extraction as feature_extraction
import mlcv.input_output as io

""" CONSTANTS """
N_JOBS = 8

def train():
    start = time.time()

    # Read the training set
    train_images_filenames, train_labels = io.load_training_set()
    print('Loaded {} train images.'.format(len(train_images_filenames)))

    # Feature extraction with sift
    print('Obtaining sift features...')
    try:
        D, L, I = io.load_object('train_sift_descriptors'), \
                  io.load_object('train_sift_labels'), \
                  io.load_object('train_sift_indices')
    except IOError:
        D, L, I = feature_extraction.parallel_sift(train_images_filenames, train_labels, num_samples_class=-1,
                                                   n_jobs=N_JOBS)
        io.save_object(D, 'train_sift_descriptors')
        io.save_object(L, 'train_sift_labels')
        io.save_object(I, 'train_sift_indices')
    print('Time spend: {:.2f} s'.format(time.time() - start))
    temp = time.time()

    # Start hyperparameters optimization
    print('\nSTARTING HYPERPARAMETER OPTIMIZATION FOR RBF SVM')
    codebook_k_values = [2 ** i for i in range(7, 16)]
    params_distribution = {
        'C': np.logspace(-4, 3, 10 ** 6),
        'gamma' : np.logspace(-3, 5, 10 ** 6)
    }
    n_iter = 100
    best_accuracy = 0
    best_params = {}
    cv_results = {}

    # Iterate codebook values
    for k in codebook_k_values:
        temp = time.time()
        print('Creating codebook with {} visual words'.format(k))
        D = D.astype(np.uint32)
        codebook = bovw.create_codebook(D, k=k, codebook_name='codebook_{}'.format(k))
        print('Time spend: {:.2f} s'.format(time.time() - temp))
        temp = time.time()

        print('Getting visual words from training set...')
        vis_words, labels = bovw.visual_words(D, L, I, codebook)
        print('Time spend: {:.2f} s'.format(time.time() - temp))
        temp = time.time()

        print('Scaling features...')
        std_scaler = StandardScaler().fit(vis_words)
        vis_words = std_scaler.transform(vis_words)
        print('Time spend: {:.2f} s'.format(time.time() - temp))
        temp = time.time()

        print('Optimizing SVM hyperparameters...')
        svm = SVC(kernel='rbf')
        random_search = RandomizedSearchCV(svm, params_distribution, n_iter=n_iter, scoring='accuracy', n_jobs=N_JOBS,
                                           refit=False, verbose=1)
        random_search.fit(vis_words, labels)
        print('Time spend: {:.2f} s'.format(time.time() - temp))
        temp = time.time()

        # Appending all parameter-scores combinations
        cv_results.update({k: random_search.cv_results_})
        io.save_object(cv_results, 'rbf_svm_optimization')

        # Obtaining the parameters which yielded the best accuracy
        if random_search.best_score_ > best_accuracy:
            best_accuracy = random_search.best_score_
            best_params = random_search.best_params_
            best_params.update({'k': k})

        print('-------------------------------\n')

    print('\nBEST PARAMS')
    print('k={}, C={} , gamma={} --> accuracy: {:.3f}'.format(best_params['k'], best_params['C'],best_params['gamma'], best_accuracy))

    print('Saving all cross-validation values...')
    io.save_object(cv_results, 'rbf_svm_optimization')
    print('Done')

def plot_curve():
    res = io.load_object('rbf_svm_optimization')
    colors = itertools.cycle(
        ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'darkolivegreen', 'darkviolet', 'black']
    )
    plt.figure()
    for k in res:
        results = res[k]
        x = results['param_C']
        y = results['mean_test_score']
        e = results['std_test_score']
        sorted_indices = x.argsort()
        x_sorted = np.asarray(x[sorted_indices], dtype=np.float64)
        y_sorted = np.asarray(y[sorted_indices], dtype=np.float64)
        e_sorted = np.asarray(e[sorted_indices], dtype=np.float64)
        color = colors.next()
        plt.errorbar(x_sorted, y_sorted, e_sorted, label='{} visual words'.format(k), color=color)

    plt.legend()
    plt.title('Optimization of C for Linear SVM')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.show()


""" MAIN SCRIPT"""
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('type', default='plot', choices=['train', 'plot'])
    args = args_parser.parse_args()
    exec_option = args.type

    if exec_option == 'train':
        train()
    else:
        plot_curve()
    exit(0)