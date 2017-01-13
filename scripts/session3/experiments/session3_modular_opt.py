from __future__ import print_function, division

import time

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.preprocessing import label_binarize
import sys

sys.path.insert(0, '.')

import mlcv.bovw as bovw
import mlcv.classification as classification
import mlcv.feature_extraction as feature_extraction
import mlcv.input_output as io
import mlcv.plotting as plotting
import mlcv.settings as settings

""" CONSTANTS """
N_JOBS = 4
K = 16


def parallel_testing(test_image, test_label, codebook, svm, scaler, pca):
    gray = io.load_grayscale_image(test_image)
    kpt, des = feature_extraction.dense(gray)
    labels = np.array([test_label] * des.shape[0])
    ind = np.array([0] * des.shape[0])
    vis_word, _ = bovw.fisher_vectors(des, labels, ind, codebook)
    prediction_prob = classification.predict_svm(vis_word, svm, std_scaler=scaler, pca=pca)
    predicted_class = svm.classes_[np.argmax(prediction_prob)]
    return predicted_class == test_label, predicted_class, np.ravel(prediction_prob)


""" MAIN SCRIPT"""
if __name__ == '__main__':
    start = time.time()
    settings.codebook_size = K
    settings.dense_sampling_density = 16
    # Read the training set
    train_images_filenames, train_labels = io.load_training_set()
    print('Loaded {} train images.'.format(len(train_images_filenames)))

    # Feature extraction with sift
    print('Obtaining dense features...')
    try:
        D, L, I = io.load_object('train_dense_descriptors_pca_60', ignore=True), \
                  io.load_object('train_dense_labels_pca_60', ignore=True), \
                  io.load_object('train_dense_indices_pca_60', ignore=True)
    except IOError:
        D, L, I, Kp = feature_extraction.parallel_sift(train_images_filenames, train_labels, num_samples_class=-1,
                                                   n_jobs=N_JOBS)
        io.save_object(D, 'train_dense_descriptors', ignore=True)
        io.save_object(L, 'train_dense_labels', ignore=True)
        io.save_object(I, 'train_dense_indices', ignore=True)

    print('Elapsed time: {:.2f} s'.format(time.time() - start))
    temp = time.time()

    print('Creating codebook with {} visual words'.format(K))
    codebook = bovw.create_gmm(D,codebook_name='gmm_{}_dense.'.format(K))
    print('Elapsed time: {:.2f} s'.format(time.time() - temp))
    temp = time.time()

    print('Getting visual words from training set...')
    vis_words, labels = bovw.fisher_vectors(D, L, I, codebook)
    print('Elapsed time: {:.2f} s'.format(time.time() - temp))
    temp = time.time()

    # Train Linear SVM classifier
    print('Training the SVM classifier...')
    print(vis_words.shape)
    lin_svm, std_scaler, pca = classification.train_linear_svm(vis_words, labels, C=1, dim_reduction=None)
    print('Elapsed time: {:.2f} s'.format(time.time() - temp))
    temp = time.time()

    # Read the test set
    test_images_filenames, test_labels = io.load_test_set()
    print('Loaded {} test images.'.format(len(test_images_filenames)))

    # Feature extraction with sift, prediction with SVM and aggregation to obtain final class
    print('Predicting test data...')
    test_results = joblib.Parallel(n_jobs=N_JOBS, backend='threading')(
        joblib.delayed(parallel_testing)(test_image, test_label, codebook, lin_svm, std_scaler, pca) for
        test_image, test_label in
        zip(test_images_filenames, test_labels))

    pred_results = [x[0] for x in test_results]
    pred_class = [x[1] for x in test_results]
    pred_prob = [x[2] for x in test_results]

    num_correct = np.count_nonzero(pred_results)
    print('Elapsed time: {:.2f} s'.format(time.time() - temp))
    temp = time.time()

    # Compute accuracy
    accuracy = num_correct * 100.0 / len(test_images_filenames)

    # Show results and timing
    print('\nACCURACY: {:.2f}'.format(accuracy))
    print('\nTOTAL TIME: {:.2f} s'.format(time.time() - start))

    classes = lin_svm.classes_

    # Create confusion matrix
    conf = confusion_matrix(test_labels, pred_class, labels=classes)

    # Create ROC curve and AUC score
    test_labels_bin = label_binarize(test_labels, classes=classes)
    fpr = []
    tpr = []
    roc_auc = []
    for i in range(len(classes)):
        c_fpr, c_tpr, _ = roc_curve(test_labels_bin[:, i], np.array(pred_prob)[:, i])
        c_roc_auc = auc(c_fpr, c_tpr)
        fpr.append(c_fpr)
        tpr.append(c_tpr)
        roc_auc.append(c_roc_auc)

    # Plot
    plotting.plot_confusion_matrix(conf, classes=classes, normalize=True)
    plotting.plot_roc_curve(fpr, tpr, roc_auc, classes=classes,
                            title='ROC curve for linear SVM with codebook of {} words'.format(K)
                            )

    print('Done.')
