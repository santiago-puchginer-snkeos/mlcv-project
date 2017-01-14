from __future__ import print_function, division

import time

import joblib

import argparse
import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import mlcv.bovw as bovw
import mlcv.classification as classification
import mlcv.feature_extraction as feature_extraction
import mlcv.input_output as io
import mlcv.plotting as plotting
import mlcv.settings as settings
import mlcv.kernels as kernels




def train():
    """ PARAMETER SWEEP """

    dense_sampling_density=[2,4,8,16]
    codebook_size=[16,32,64]
    pca_reduction=[60,80,100,120]
    params_distribution = {
        'C': np.logspace(-3, 1, 10 ** 6)
    }
    n_iter = 20
    best_accuracy = 0
    best_params = {}
    cv_results = {}
    """ SETTINGS """
    settings.n_jobs = 1
    #settings.codebook_size = 16
    #settings.dense_sampling_density = 16
    #settings.pca_reduction = 64

    start = time.time()

    # Read the training set
    train_images_filenames, train_labels = io.load_training_set()
    print('Loaded {} train images.'.format(len(train_images_filenames)))

    # Parameter sweep for dense SIFT
    for ds in (dense_sampling_density):
        settings.dense_sampling_density = ds
        # Feature extraction with sift
        print('Obtaining dense features...')
        # try:
        #     D, L, I = io.load_object('train_dense_descriptors_{}'.format(settings.dense_sampling_density), ignore=True), \
        #               io.load_object('train_dense_labels_{}'.format(settings.dense_sampling_density), ignore=True), \
        #               io.load_object('train_dense_indices_{}'.format(settings.dense_sampling_density), ignore=True)
        # except IOError:
        D, L, I, _ = feature_extraction.parallel_dense(train_images_filenames, train_labels,
                                                           num_samples_class=-1,
                                                           n_jobs=settings.n_jobs)
        #     io.save_object(D, 'train_dense_descriptors_{}'.format(settings.dense_sampling_density), ignore=True)
        #     io.save_object(L, 'train_dense_labels_{}'.format(settings.dense_sampling_density), ignore=True)
        #     io.save_object(I, 'train_dense_indices_{}'.format(settings.dense_sampling_density), ignore=True)
        print('Elapsed time: {:.2f} s'.format(time.time() - start))

        elapsed_sift=time.time()-start

        # Parameter sweep for PCA
        for dim_red in pca_reduction:
            temp1 = time.time()
            settings.pca_reduction = dim_red

            print('Applying PCA...')
            pca, D = feature_extraction.pca(D)
            print('Elapsed time: {:.2f} s'.format(time.time() - temp1))
            elapsed_pca=time.time()-temp1

        #Parameter sweep for codebook size
            for k in codebook_size:
                temp2 = time.time()
                settings.codebook_size = k

                print('Creating codebook with {} visual words'.format(settings.codebook_size))
                # gmm = bovw.create_gmm(D, codebook_name='gmm_{}_dense'.format(settings.codebook_size))
                gmm = bovw.create_gmm(D, None)
                print('Elapsed time: {:.2f} s'.format(time.time() - temp))
                temp = time.time()

                print('Getting visual words from training set...')
                fisher, labels = bovw.fisher_vectors(D, L, I, gmm, normalization='power')
                print('Elapsed time: {:.2f} s'.format(time.time() - temp))
                temp = time.time()

                # Train Linear SVM classifier
                print('Training the SVM classifier...')


                print('Scaling features...')
                std_scaler = StandardScaler().fit(fisher)
                vis_words = std_scaler.transform(fisher)
                print('Elapsed time: {:.2f} s'.format(time.time() - temp))
                temp = time.time()

                print('Optimizing SVM hyperparameters...')
                svm = SVC(kernel='precomputed')
                random_search = RandomizedSearchCV(svm, params_distribution, n_iter=n_iter, scoring='accuracy',
                                                   n_jobs=settings.n_jobs,
                                                   refit=False, verbose=1, cv=4)
                # Precompute Gram matrix
                gram = kernels.intersection_kernel(vis_words, vis_words)
                random_search.fit(gram, labels)
                print('Elapsed time: {:.2f} s'.format(time.time() - temp))
                elapsed_cv=time.time()
                total_time=elapsed_sift+elapsed_pca+elapsed_cv
                # Convert MaskedArrays to ndarrays to avoid unpickling bugs
                results = random_search.cv_results_
                results['param_C'] = results['param_C'].data

                # Appending all parameter-scores combinations
                cv_results.update({[k,dim_red,ds,total_time]: results})
                io.save_object(cv_results, 'intersection_svm_optimization_fisher_vectors_power')

                # Obtaining the parameters which yielded the best accuracy
                if random_search.best_score_ > best_accuracy:
                    best_accuracy = random_search.best_score_
                    best_params = random_search.best_params_
                    best_params.update({'k': k, 'pca': dim_red, 'dense_grid': ds})

                print('-------------------------------\n')

                print('Elapsed time: {:.2f} s'.format(time.time() - temp))
                temp = time.time()

    print('\nBEST PARAMS')
    print('k={}, C={} --> accuracy: {:.3f}'.format(best_params['k'], best_params['C'], best_accuracy))

    print('Saving all cross-validation values...')
    io.save_object(cv_results, 'intersection_svm_optimization_fisher_vectors_power')
    print('Done')



def plot_curve():
    print('Loading results object...')
    res = io.load_object('intersection_svm_optimization', ignore=True)

    print('Plotting...')
    colors = itertools.cycle(
        ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'darkolivegreen', 'darkviolet', 'black']
    )
    plt.figure(figsize=(20, 10), dpi=200, facecolor='white')
    # Compute subplot parameters
    num_subplots = len(res)
    num_rows = np.ceil(num_subplots / 2)
    # All subplots
    for ind, k in enumerate(sorted(res.keys())):
        # Plot
        results = res[k]
        x = results['param_C']
        y = results['mean_test_score']
        e = results['std_test_score']
        sorted_indices = x.argsort()
        x_sorted = np.asarray(x[sorted_indices], dtype=np.float64)
        y_sorted = np.asarray(y[sorted_indices], dtype=np.float64)
        e_sorted = np.asarray(e[sorted_indices], dtype=np.float64)
        color = colors.next()
        ax = plt.subplot(num_rows, 2, ind + 1)
        ax.set_xscale("log")
        ax.errorbar(x_sorted, y_sorted, e_sorted, linestyle='--', lw=2, marker='x', color=color)
        ax.set_title('{} visual words'.format(k))
        ax.set_ylim((0.25, 0.6))
        ax.set_xlabel('C')
        ax.set_ylabel('Accuracy')

        # Print information
        print('CODEBOOK {} '.format(k))
        print('-------------')
        print('Mean accuracy: {}'.format(y.max()))
        print('Std accuracy: {}'.format(e[np.argmax(y)]))
        print('C: {}'.format(x[np.argmax(y)]))
        print()
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
