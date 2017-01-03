from __future__ import print_function, division

import argparse
import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
        D, L, I = io.load_object('train_sift_descriptors', ignore=True), \
                  io.load_object('train_sift_labels', ignore=True), \
                  io.load_object('train_sift_indices', ignore=True)
    except IOError:
        D, L, I = feature_extraction.parallel_sift(train_images_filenames, train_labels, num_samples_class=-1,
                                                   n_jobs=N_JOBS)
        io.save_object(D, 'train_sift_descriptors', ignore=True)
        io.save_object(L, 'train_sift_labels', ignore=True)
        io.save_object(I, 'train_sift_indices', ignore=True)
    print('Elapsed time: {:.2f} s'.format(time.time() - start))

    # Start hyperparameters optimization
    print('\nSTARTING HYPERPARAMETER OPTIMIZATION FOR LINEAR SVM')
    codebook_k_values = [2 ** i for i in range(7, 16)]
    params_distribution = {
        'C': np.logspace(-4, 3, 10 ** 6)
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
        print('Elapsed time: {:.2f} s'.format(time.time() - temp))
        temp = time.time()

        print('Getting visual words from training set...')
        vis_words, labels = bovw.visual_words(D, L, I, codebook)
        print('Elapsed time: {:.2f} s'.format(time.time() - temp))
        temp = time.time()

        print('Scaling features...')
        std_scaler = StandardScaler().fit(vis_words)
        vis_words = std_scaler.transform(vis_words)
        print('Elapsed time: {:.2f} s'.format(time.time() - temp))
        temp = time.time()

        print('Optimizing SVM hyperparameters...')
        svm = SVC(kernel='linear')
        random_search = RandomizedSearchCV(svm, params_distribution, n_iter=n_iter, scoring='accuracy', n_jobs=N_JOBS,
                                           refit=False, verbose=1, cv=4)
        random_search.fit(vis_words, labels)
        print('Elapsed time: {:.2f} s'.format(time.time() - temp))

        # Appending all parameter-scores combinations
        cv_results.update({k: random_search.cv_results_})
        io.save_object(cv_results, 'linear_svm_optimization')

        # Obtaining the parameters which yielded the best accuracy
        if random_search.best_score_ > best_accuracy:
            best_accuracy = random_search.best_score_
            best_params = random_search.best_params_
            best_params.update({'k': k})

        print('-------------------------------\n')

    print('\nBEST PARAMS')
    print('k={}, C={} --> accuracy: {:.3f}'.format(best_params['k'], best_params['C'], best_accuracy))

    print('Saving all cross-validation values...')
    io.save_object(cv_results, 'linear_svm_optimization')
    print('Done')


def plot_curve():
    print('Loading results object...')
    res = io.load_object('linear_svm_optimization', ignore=True)

    print('Plotting...')
    colors = itertools.cycle(
        ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'darkolivegreen', 'darkviolet', 'black']
    )
    plt.figure(figsize=(40, 20), facecolor='white')
    # Compute subplot parameters
    num_subplots = len(res)
    num_rows = np.ceil(num_subplots / 2)
    # All subplots
    for ind, k in enumerate(sorted(res.keys())):
        results = res[k]
        x = results['param_C']
        y = results['mean_test_score']
        e = results['std_test_score']
        sorted_indices = x.argsort()
        x_sorted = np.asarray(x[sorted_indices], dtype=np.float64)
        y_sorted = np.asarray(y[sorted_indices], dtype=np.float64)
        e_sorted = np.asarray(e[sorted_indices], dtype=np.float64)
        color = colors.next()
        ax = plt.subplot(num_rows, 2, ind+1)
        ax.set_xscale("log")
        ax.errorbar(x_sorted, y_sorted, e_sorted, linestyle='--', lw=2, marker='x', color=color)
        ax.set_title('{} visual words'.format(k))
        ax.set_ylim((0.25, 0.6))
        ax.set_xlabel('C')
        ax.set_ylabel('Accuracy')
    plt.tight_layout()
    plt.show()
    plt.close()


""" MAIN SCRIPT"""
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--type', default='train', choices=['train', 'plot'])
    args = args_parser.parse_args()
    exec_option = args.type

    if exec_option == 'train':
        train()
    elif exec_option == 'plot':
        plot_curve()
    exit(0)
