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


def parallel_testing(test_image, test_label, svm, scaler, gmm, pca):
    gray = io.load_grayscale_image(test_image)
    kpt, des = feature_extraction.dense(gray)
    labels = np.array([test_label] * des.shape[0])
    ind = np.array([0] * des.shape[0])

    des = pca.transform(des)
    fisher, _ = bovw.fisher_vectors(des, labels, ind, gmm)

    prediction_prob = classification.predict_svm(fisher, svm, std_scaler=scaler)
    predicted_class = svm.classes_[np.argmax(prediction_prob)]
    return predicted_class == test_label, predicted_class, np.ravel(prediction_prob)


""" MAIN SCRIPT"""
if __name__ == '__main__':

    """ SETTINGS """
    settings.n_jobs = 1
    plot_results = 1
    calculate_results = 0

    # Optimal parameters
    C = 2.1539
    settings.codebook_size = 64
    settings.dense_sampling_density = 8
    settings.pca_reduction = 60

    if calculate_results == 1:
        start = time.time()

        # Read the training set
        train_images_filenames, train_labels = io.load_training_set()
        print('Loaded {} train images.'.format(len(train_images_filenames)))

        # Feature extraction with sift
        print('Obtaining dense features...')
        try:
            D, L, I = io.load_object('train_dense_descriptors', ignore=True), \
                      io.load_object('train_dense_labels', ignore=True), \
                      io.load_object('train_dense_indices', ignore=True)
        except IOError:
            D, L, I, _ = feature_extraction.parallel_dense(train_images_filenames, train_labels,
                                                           num_samples_class=-1,
                                                           n_jobs=settings.n_jobs)
            io.save_object(D, 'train_dense_descriptors', ignore=True)
            io.save_object(L, 'train_dense_labels', ignore=True)
            io.save_object(I, 'train_dense_indices', ignore=True)
        print('Elapsed time: {:.2f} s'.format(time.time() - start))
        temp = time.time()

        print('Applying PCA...')
        pca, D = feature_extraction.pca(D)
        print('Elapsed time: {:.2f} s'.format(time.time() - temp))
        temp = time.time()

        print('Creating GMM model with {} Gaussians'.format(settings.codebook_size))
        gmm = bovw.create_gmm(D, codebook_name='gmm_{}_dense_pca_{}'.format(settings.codebook_size, settings.pca_reduction))
        print('Elapsed time: {:.2f} s'.format(time.time() - temp))
        temp = time.time()

        print('Getting Fisher vectors from training set...')
        fisher, labels = bovw.fisher_vectors(D, L, I, gmm)
        print('Elapsed time: {:.2f} s'.format(time.time() - temp))
        temp = time.time()

        # Train Linear SVM classifier
        print('Training the SVM classifier...')
        svm, std_scaler, _ = classification.train_intersection_svm(fisher, labels, C=C, dim_reduction=None)
        print('Elapsed time: {:.2f} s'.format(time.time() - temp))
        temp = time.time()

        # Read the test set
        test_images_filenames, test_labels = io.load_test_set()
        print('Loaded {} test images.'.format(len(test_images_filenames)))

        # Feature extraction with sift, prediction with SVM and aggregation to obtain final class
        print('Predicting test data...')
        test_results = joblib.Parallel(n_jobs=settings.n_jobs, backend='threading')(
            joblib.delayed(parallel_testing)(test_image, test_label, svm, std_scaler, gmm, pca) for
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

        classes = svm.classes_

        io.save_object(test_results, 'fisher_pca_intersection_test_results')
        io.save_object(classes, 'fisher_pca_intersection_classes')

    if plot_results == 1:
        print('Loading results object...')
        test_results = io.load_object('fisher_pca_intersection_test_results', ignore=True)
        classes = io.load_object('fisher_pca_intersection_classes', ignore=True)
        pred_results = [x[0] for x in test_results]
        pred_class = [x[1] for x in test_results]
        pred_prob = [x[2] for x in test_results]

        test_images_filenames, test_labels = io.load_test_set()

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
