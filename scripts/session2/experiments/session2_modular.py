from __future__ import print_function, division

import time

import numpy as np

import mlcv.bovw as bovw
import mlcv.classification as classification
import mlcv.feature_extraction as feature_extraction
import mlcv.input_output as io

""" CONSTANTS """
K = 512


def seq_testing(test_image, test_label, codebook, svm, scaler, pca):
    gray = io.load_grayscale_image(test_image)
    kpt, des = feature_extraction.sift(gray)
    labels = np.array([test_label] * des.shape[0])
    ind = np.array([0] * des.shape[0])
    vis_word, _ = bovw.visual_words(des, labels, ind, codebook)
    predictions = classification.predict_svm(vis_word, svm, std_scaler=scaler, pca=pca)
    predicted_class = predictions[0]
    return predicted_class == test_label


""" MAIN SCRIPT"""
if __name__ == '__main__':
    start = time.time()

    # Read the training set
    train_images_filenames, train_labels = io.load_training_set()
    print('Loaded {} train images.'.format(len(train_images_filenames)))

    # Feature extraction with sift
    print('Obtaining sift features...')
    D, L, I = feature_extraction.seq_sift(train_images_filenames, train_labels, num_samples_class=-1)
    print('Time spend: {:.2f} s'.format(time.time() - start))
    temp = time.time()

    print('Creating codebook with {} visual words'.format(K))
    codebook = bovw.create_codebook(D, k=K, codebook_name='default_codebook')
    print('Time spend: {:.2f} s'.format(time.time() - temp))
    temp = time.time()

    print('Getting visual words from training set...')
    vis_words, labels = bovw.visual_words(D, L, I, codebook)
    print('Time spend: {:.2f} s'.format(time.time() - temp))
    temp = time.time()

    # Train Linear SVM classifier
    print('Training the SVM classifier...')
    lin_svm, std_scaler, pca = classification.train_linear_svm(vis_words, labels, C=1, dim_reduction=None)
    print('Time spend: {:.2f} s'.format(time.time() - temp))
    temp = time.time()

    # Read the test set
    test_images_filenames, test_labels = io.load_test_set()
    print('Loaded {} test images.'.format(len(test_images_filenames)))

    # Feature extraction with sift, prediction with SVM and aggregation to obtain final class
    print('Predicting test data...')
    correct_class = [seq_testing(test_image, test_label, codebook, lin_svm, std_scaler, pca) for
                     test_image, test_label in
                     zip(test_images_filenames, test_labels)]
    num_correct = np.count_nonzero(correct_class)
    print('Time spend: {:.2f} s'.format(time.time() - temp))
    temp = time.time()

    # Compute accuracy
    accuracy = num_correct * 100.0 / len(test_images_filenames)

    # Show results and timing
    print('\nACCURACY: {:.2f}'.format(accuracy))
    print('\nTOTAL TIME: {:.2f} s'.format(time.time() - start))
