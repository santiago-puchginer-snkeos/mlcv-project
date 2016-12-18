from __future__ import print_function, division

import time

import joblib
import numpy as np

import mlcv.classification as classification
import mlcv.feature_extraction as feature_extraction
import mlcv.io as io

""" CONSTANTS """
N_JOBS = 6


def parallel_testing(test_image, test_label, lin_svm, std_scaler):
    gray = io.load_grayscale_image(test_image)
    kpt, des = feature_extraction.sift(gray)
    predictions = classification.predict_svm(des, lin_svm, std_scaler=std_scaler)
    values, counts = np.unique(predictions, return_counts=True)
    predicted_class = values[np.argmax(counts)]
    return predicted_class == test_label


""" MAIN SCRIPT"""
if __name__ == '__main__':
    start = time.time()

    # Read the training set
    train_images_filenames, train_labels = io.load_training_set()
    print('Loaded {} train images.'.format(len(train_images_filenames)))

    # Feature extraction with sift
    print('Obtaining sift features...')
    D, L, _ = feature_extraction.parallel_sift(train_images_filenames, train_labels, num_samples_classes=30,
                                               n_jobs=N_JOBS)
    print('Time spend: {:.2f} s'.format(time.time() - start))
    temp = time.time()

    # Train Linear SVM classifier
    print('Training the SVM classifier...')
    lin_svm, std_scaler, pca = classification.train_linear_svm(D, L, model_name='linsvm_sift_30s')
    print('Time spend: {:.2f} s'.format(time.time() - temp))
    temp = time.time()

    # Read the test set
    test_images_filenames, test_labels = io.load_test_set()
    print('Loaded {} test images.'.format(len(test_images_filenames)))

    # Feature extraction with sift, prediction with SVM and aggregation to obtain final class
    print('Predicting test data...')
    correct_class = joblib.Parallel(n_jobs=N_JOBS, backend='threading')(
        joblib.delayed(parallel_testing)(test_image, test_label, lin_svm, std_scaler) for test_image, test_label in
        zip(test_images_filenames, test_labels))
    num_correct = np.count_nonzero(correct_class)
    print('Time spend: {:.2f} s'.format(time.time() - temp))
    temp = time.time()

    # Compute accuracy
    accuracy = num_correct * 100.0 / len(test_images_filenames)

    # Show results and timing
    print('\nACCURACY: {:.2f}'.format(accuracy))
    print('\nTOTAL TIME: {:.2f} s'.format(time.time() - start))
