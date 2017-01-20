from __future__ import print_function, division

import argparse
import itertools
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import mlcv.bovw as bovw
import mlcv.feature_extraction as feature_extraction
import mlcv.input_output as io
import mlcv.kernels as kernels
import mlcv.settings as settings

""" PARAMETER SWEEP """
dense_sampling_density = [8, 16]
codebook_size = [16, 32, 64]
params_distribution = {
    'C': np.logspace(-5, 3, 10 ** 6)
}
n_iter = 50


def train():
    best_accuracy = 0
    best_params = {}
    cv_results = {}

    """ SETTINGS """
    settings.n_jobs = 1

    # Read the training set
    train_images_filenames, train_labels = io.load_training_set()
    io.log('Loaded {} train images.'.format(len(train_images_filenames)))

    # Parameter sweep for dense SIFT
    for ds in dense_sampling_density:

        io.log('Obtaining dense features with sampling parameter {}...'.format(ds))
        start_sift = time.time()
        settings.dense_sampling_density = ds
        try:
            D, L, I = io.load_object('train_dense_descriptors_{}'.format(settings.dense_sampling_density), ignore=True), \
                      io.load_object('train_dense_labels_{}'.format(settings.dense_sampling_density), ignore=True), \
                      io.load_object('train_dense_indices_{}'.format(settings.dense_sampling_density), ignore=True)
        except IOError:
            D, L, I, _ = feature_extraction.parallel_dense(train_images_filenames, train_labels,
                                                           num_samples_class=-1,
                                                           n_jobs=settings.n_jobs)
            io.save_object(D, 'train_dense_descriptors_{}'.format(settings.dense_sampling_density), ignore=True)
            io.save_object(L, 'train_dense_labels_{}'.format(settings.dense_sampling_density), ignore=True)
            io.save_object(I, 'train_dense_indices_{}'.format(settings.dense_sampling_density), ignore=True)
        sift_time = time.time() - start_sift
        io.log('Elapsed time: {:.2f} s'.format(sift_time))

        # Parameter sweep for codebook size
        for k in codebook_size:

            io.log('Creating GMM model (k = {})'.format(k))
            start_gmm = time.time()
            settings.codebook_size = k
            gmm = bovw.create_gmm(D, 'gmm_{}_dense_{}'.format(
                k,
                ds,
            ))
            gmm_time = time.time() - start_gmm
            io.log('Elapsed time: {:.2f} s'.format(gmm_time))

            io.log('Getting Fisher vectors from training set...')
            start_fisher = time.time()
            fisher, labels = bovw.fisher_vectors(D, L, I, gmm, normalization='power')
            fisher_time = time.time() - start_fisher
            io.log('Elapsed time: {:.2f} s'.format(fisher_time))

            io.log('Scaling features...')
            start_scaler = time.time()
            std_scaler = StandardScaler().fit(fisher)
            vis_words = std_scaler.transform(fisher)
            scaler_time = time.time() - start_scaler
            io.log('Elapsed time: {:.2f} s'.format(scaler_time))

            io.log('Optimizing SVM hyperparameters...')
            start_crossvalidation = time.time()
            svm = SVC(kernel='precomputed')
            random_search = RandomizedSearchCV(
                svm,
                params_distribution,
                n_iter=n_iter,
                scoring='accuracy',
                n_jobs=settings.n_jobs,
                refit=False,
                cv=3,
                verbose=1
            )
            # Precompute Gram matrix
            gram = kernels.intersection_kernel(vis_words, vis_words)
            random_search.fit(gram, labels)
            crossvalidation_time = time.time() - start_crossvalidation
            io.log('Elapsed time: {:.2f} s'.format(crossvalidation_time))

            # Convert MaskedArrays to ndarrays to avoid unpickling bugs
            results = random_search.cv_results_
            results['param_C'] = results['param_C'].data

            # Appending all parameter-scores combinations
            cv_results.update({
                (k, ds): {
                    'cv_results': results,
                    'sift_time': sift_time,
                    'gmm_time': gmm_time,
                    'fisher_time': fisher_time,
                    'scaler_time': scaler_time,
                    'crossvalidation_time': crossvalidation_time,
                    'total_time': sift_time + gmm_time + fisher_time + scaler_time + crossvalidation_time
                }
            })
            io.save_object(cv_results, 'intersection_svm_optimization_fisher_vectors_power_no_pca', ignore=True)

            # Obtaining the parameters which yielded the best accuracy
            if random_search.best_score_ > best_accuracy:
                best_accuracy = random_search.best_score_
                best_params = random_search.best_params_
                best_params.update({'k': k, 'ds': ds})

            io.log('-------------------------------\n')

    io.log('\nSaving best parameters...')
    io.save_object(best_params, 'best_params_intersection_svm_optimization_fisher_vectors_power_no_pca', ignore=True)
    best_params_file = os.path.abspath(
        './ignore/best_params_intersection_svm_optimization_fisher_vectors_power_no_pca.pickle')
    io.log('Saved at {}'.format(best_params_file))

    io.log('\nSaving all cross-validation values...')
    io.save_object(cv_results, 'intersection_svm_optimization_fisher_vectors_power_no_pca', ignore=True)
    cv_results_file = os.path.abspath('./ignore/intersection_svm_optimization_fisher_vectors_power_no_pca.pickle')
    io.log('Saved at {}'.format(cv_results_file))

    io.log('\nBEST PARAMS')
    io.log('k={}, C={}, dense_grid={} --> accuracy: {:.3f}'.format(
        best_params['k'],
        best_params['C'],
        best_params['ds'],
        best_accuracy
    ))


def plot_curve():
    io.log('Loading cross-validation values...')
    cv_values = io.load_object('intersection_svm_optimization_fisher_vectors_power_no_pca', ignore=True)

    io.log('Loading best parameters...')
    best_params = io.load_object('best_params_intersection_svm_optimization_fisher_vectors_power_no_pca', ignore=True)

    io.log('Plotting...')
    colors = itertools.cycle(
        ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'darkolivegreen', 'darkviolet', 'black']
    )

    best_ds = best_params['ds']

    io.log()
    io.log('BEST PARAMETERS')
    io.log('Dense sampling = {}'.format(best_ds))
    io.log()

    # Subplot parameters
    plt.figure(figsize=(20, 10), dpi=200, facecolor='white')
    num_subplots = len(codebook_size)
    num_columns = 2
    num_rows = np.ceil(num_subplots / num_columns)

    # All subplots
    for ind, k in enumerate(codebook_size):
        # Search dictionary
        val = cv_values[(k, best_ds)]
        results = val['cv_results']
        sift_time = val['sift_time']
        gmm_time = val['gmm_time']
        fisher_time = val['fisher_time']
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
        ax = plt.subplot(num_rows, num_columns, ind + 1)
        ax.set_xscale("log")
        ax.set_ylim((0.6, 0.9))
        ax.errorbar(x_sorted, y_sorted, e_sorted, linestyle='--', lw=2, marker='x', color=color)
        ax.set_title('{} Gaussians in GMM'.format(k))
        ax.set_xlabel('C')
        ax.set_ylabel('Accuracy')

        # Print information
        io.log('CODEBOOK {} '.format(k))
        io.log('-------------')
        io.log('Mean accuracy: {}'.format(y.max()))
        io.log('Std accuracy: {}'.format(e[np.argmax(y)]))
        io.log('C: {}'.format(x[np.argmax(y)]))
        io.log()
        io.log('Timing')
        io.log('\tSIFT time: {:.2f} s'.format(sift_time))
        io.log('\tGMM time: {:.2f} s'.format(gmm_time))
        io.log('\tFisher time: {:.2f} s'.format(fisher_time))
        io.log('\tScaler time: {:.2f} s'.format(scaler_time))
        io.log('\tCV time: {:.2f} s'.format(crossvalidation_time))
        io.log('\t_________________________')
        io.log('\tTOTAL TIME: {:.2f} s'.format(total_time))
        io.log()
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
