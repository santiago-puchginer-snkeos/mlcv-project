from __future__ import print_function, division

import time

import joblib
import numpy as np
from sklearn import metrics

import mlcv.classification as classification
import mlcv.feature_extraction as feature_extraction
import mlcv.input_output as io
from mlcv.plotting import plot_confusion_matrix
from scripts import SESSION1


def parallel_testing(test_image, test_label, svm, std_scaler, pca):
    gray = io.load_grayscale_image(test_image)
    kpt, des = feature_extraction.sift(gray)
    predictions = classification.predict_svm(des, svm, std_scaler=std_scaler, pca=pca, probability=True)
    probabilities = np.sum(predictions, axis=0)
    predicted_class = svm.classes_[np.argmax(probabilities)]

    return predicted_class == test_label, predicted_class, test_label


if __name__ == '__main__':
    start = time.time()

    # Read the test set
    test_images_filenames, test_labels = io.load_test_set()
    print('Loaded {} test images.'.format(len(test_images_filenames)))

    # Load the trained model, scaler and PCA
    svm = io.load_object(SESSION1['model'])
    std_scaler = io.load_object(SESSION1['scaler'])
    pca = io.load_object(SESSION1['pca'])

    # Feature extraction with sift, prediction with SVM and aggregation to obtain final class
    print('Predicting test data...')
    result = joblib.Parallel(n_jobs=SESSION1['n_jobs'], backend='threading')(
        joblib.delayed(parallel_testing)(
            test_image,
            test_label,
            svm,
            std_scaler,
            pca
        ) for test_image, test_label in
        zip(test_images_filenames, test_labels))

    correct_class = [i[0] for i in result]
    predicted = [i[1] for i in result]
    expected = [i[2] for i in result]

    num_correct = np.count_nonzero(correct_class)
    print('Time spend: {:.2f} s'.format(time.time() - start))
    temp = time.time()

    # Compute accuracy
    accuracy = num_correct * 100.0 / len(test_images_filenames)

    # Plot and save normalized confusion matrix
    conf = metrics.confusion_matrix(expected, predicted, labels=svm.classes_)
    plot_confusion_matrix(conf, classes=svm.classes_, normalize=True)
    io.save_object(conf, SESSION1['conf_matrix'])

    # Show results and timing
    print('\nACCURACY: {:.2f}'.format(accuracy))
    print('\nTOTAL TIME: {:.2f} s'.format(time.time() - start))
