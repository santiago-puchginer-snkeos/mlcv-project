from __future__ import print_function, division

import time

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

import mlcv.bovw as bovw
import mlcv.feature_extraction as feature_extraction
import mlcv.input_output as io

""" CONSTANTS """
N_JOBS = 8

""" MAIN SCRIPT"""
if __name__ == '__main__':
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
    print('\nSTARTING HYPERPARAMETER OPTIMIZATION FOR LINEAR SVM')
    codebook_k_values = [2 ** i for i in range(6, 16)]
    params_distribution = {
        'C': np.logspace(-1, 3, 10 ** 6)
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

        # Optimizing SVM hyperparameters
        print('Optimizing SVM hyperparameters...')
        svm = SVC(kernel='linear')
        random_search = RandomizedSearchCV(svm, params_distribution, n_iter=n_iter, scoring='accuracy', n_jobs=N_JOBS,
                                           refit=False, verbose=1)
        random_search.fit(vis_words, labels)
        print('Time spend: {:.2f} s'.format(time.time() - temp))
        temp = time.time()

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
