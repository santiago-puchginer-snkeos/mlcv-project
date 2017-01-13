from __future__ import print_function, division

import time

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.preprocessing import label_binarize


import mlcv.bovw as bovw
import mlcv.classification as classification
import mlcv.feature_extraction as feature_extraction
import mlcv.input_output as io
import mlcv.plotting as plotting
import mlcv.settings as settings

import sklearn.decomposition as decomposition
from yael import ynumpy

""" CONSTANTS """
N_JOBS = 4
settings.codebook_size = 16
settings.dense_sampling_density = 16
settings.pca_reduction = 60

def parallel_testing(test_image, test_label, svm, scaler,gmm):
    fisher_test = np.zeros((1, settings.codebook_size * 128 * 2), dtype=np.float32)
    gray = io.load_grayscale_image(test_image)
    kpt, des = feature_extraction.dense(gray)
    labels = np.array([test_label] * des.shape[0])
    ind = np.array([0] * des.shape[0])
 #   pca = decomposition.PCA(n_components=settings.pca_reduction)
 #   pca.fit(des)
 #   des = pca.transform(des)
   # vis_word, _ = bovw.fisher_vectors(des, labels, ind, codebook)
    fisher_test = ynumpy.fisher(gmm, des, include=['mu', 'sigma'])


    prediction_prob = classification.predict_svm(fisher_test, svm, std_scaler=scaler)
    predicted_class = svm.classes_[np.argmax(prediction_prob)]
    return predicted_class == test_label, predicted_class, np.ravel(prediction_prob)


train_descriptors = []
""" MAIN SCRIPT"""
if __name__ == '__main__':
    start = time.time()

    # Read the training set
    train_images_filenames, train_labels = io.load_training_set()
    print('Loaded {} train images.'.format(len(train_images_filenames)))

    # Feature extraction with sift
    print('Obtaining dense features...')
    try:
        D, L, I, train_descriptors = io.load_object('train_dense_16_descriptors', ignore=True), \
                  io.load_object('train_dense_16_labels', ignore=True), \
                  io.load_object('train_dense_16_indices', ignore=True), \
                  io.load_object('train_dense_16_descriptors_list', ignore=True)
    except IOError:
        D, L, I, Kp,train_descriptors = feature_extraction.parallel_dense(train_images_filenames, train_labels, num_samples_class=-1,
                                                   n_jobs=N_JOBS)
        io.save_object(D, 'train_dense_16_descriptors', ignore=True)
        io.save_object(train_descriptors, 'train_dense_16_descriptors_list', ignore=True)
        io.save_object(L, 'train_dense_16_labels', ignore=True)
        io.save_object(I, 'train_dense_16_indices', ignore=True)

    print('Elapsed time: {:.2f} s'.format(time.time() - start))
    temp = time.time()

    print('Applying PCA...')
 #   pca = decomposition.PCA(n_components=settings.pca_reduction)
 #   pca.fit(D)
 #   D = pca.transform(D)

    print('Elapsed time: {:.2f} s'.format(time.time() - temp))
    temp = time.time()

    print('Creating codebook with {} visual words'.format(settings.codebook_size))
    #codebook = bovw.create_gmm(D,codebook_name='gmm_{}_dense.'.format(settings.codebook_size))
    gmm = ynumpy.gmm_learn(np.float32(D), settings.codebook_size)
    print('Elapsed time: {:.2f} s'.format(time.time() - temp))
    temp = time.time()

    print('Getting visual words from training set...')
    #vis_words, labels = bovw.fisher_vectors(D, L, I, codebook)

    fisher = np.zeros((len(train_descriptors), settings.codebook_size * 128 * 2), dtype=np.float32)
    for i in xrange(len(train_descriptors)):
        fisher[i, :] = ynumpy.fisher(gmm, train_descriptors[i], include=['mu', 'sigma'])

    print('Elapsed time: {:.2f} s'.format(time.time() - temp))

    # Train Linear SVM classifier
    print('Training the SVM classifier...')


    lin_svm, std_scaler, _ = classification.train_linear_svm(fisher, L, C=1,dim_reduction=None)
    print('Elapsed time: {:.2f} s'.format(time.time() - temp))
    temp = time.time()

    # Read the test set
    test_images_filenames, test_labels = io.load_test_set()
    print('Loaded {} test images.'.format(len(test_images_filenames)))

    # Feature extraction with sift, prediction with SVM and aggregation to obtain final class
    print('Predicting test data...')
    test_results = joblib.Parallel(n_jobs=N_JOBS, backend='threading')(
        joblib.delayed(parallel_testing)(test_image, test_label, lin_svm, gmm) for
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
                            title='ROC curve for linear SVM with codebook of {} words'.format(settings.codebook_size)
                            )

    print('Done.')
