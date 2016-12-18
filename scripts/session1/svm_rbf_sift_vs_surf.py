from __future__ import print_function, division

import time

import joblib
import numpy as np
import sklearn.metrics as metrics

import mlcv.classification as classification
import mlcv.feature_extraction as feature_extraction
import mlcv.input_output as io
from mlcv.plotting import plot_confusion_matrix


def parallel_testing_sift(test_image, test_label, svm, std_scaler, pca):
    gray = io.load_grayscale_image(test_image)
    kpt, des = feature_extraction.sift(gray)
    if des is not None:
        predictions = classification.predict_svm(des, svm, std_scaler=std_scaler, pca=pca, probability=True)
        probabilities = np.sum(predictions, axis=0)
        predicted_class = svm.classes_[np.argmax(probabilities)]
        return predicted_class == test_label, predicted_class, test_label
    else:
        return False, np.nan, test_label


def parallel_testing_surf(test_image, test_label, svm, std_scaler, pca):
    gray = io.load_grayscale_image(test_image)
    kpt, des = feature_extraction.surf(gray)
    if des is not None:
        predictions = classification.predict_svm(des, svm, std_scaler=std_scaler, pca=pca, probability=True)
        probabilities = np.sum(predictions, axis=0)
        predicted_class = svm.classes_[np.argmax(probabilities)]
        return predicted_class == test_label, predicted_class, test_label
    else:
        return False, np.nan, test_label


""" CONSTANTS """
N_JOBS = 6
DIM_REDUCTION_OPTIONS = [23, None]
FEATURE_EXTRACTION_OPTIONS = {
    'sift': (feature_extraction.parallel_sift, parallel_testing_sift),
    'surf': (feature_extraction.parallel_surf, parallel_testing_surf)
}
NUM_SAMPLES_OPTIONS = [-1, 30]

""" MAIN SCRIPT"""
if __name__ == '__main__':
    start = time.time()

    # Read the training set
    train_images_filenames, train_labels = io.load_training_set()
    print('Loaded {} train images.'.format(len(train_images_filenames)))

    # Read the test set
    test_images_filenames, test_labels = io.load_test_set()
    print('Loaded {} test images.'.format(len(test_images_filenames)))

    # Iterate over all options
    results = []
    for dim_red_option in DIM_REDUCTION_OPTIONS:
        for num_samples in NUM_SAMPLES_OPTIONS:
            for fe_name, fe_functions in FEATURE_EXTRACTION_OPTIONS.iteritems():
                # Progress report
                print('\nFEATURE EXTRACTOR: {}'.format(fe_name))
                print('NUMBER OF SAMPLES PER CLASS: {}'.format(num_samples if num_samples > -1 else 'ALL'))
                print('PCA COMPONENTS: {}'.format(dim_red_option if dim_red_option is not None else 'ALL (NO PCA)'))

                # Unpack train and prediction feature extractors
                train_function = fe_functions[0]
                predict_function = fe_functions[1]

                # Feature extraction
                print('Obtaining features...')
                D, L, _ = train_function(train_images_filenames, train_labels, num_samples_class=num_samples,
                                         n_jobs=N_JOBS)
                print('Time spend: {:.2f} s'.format(time.time() - start))
                temp = time.time()

                # Train Linear SVM classifier
                print('Training the SVM with RBF kernel classifier...')
                svm, std_scaler, pca = classification.train_rbf_svm(
                    D,
                    L,
                    C=5,
                    gamma=0.1,
                    dim_reduction=dim_red_option,
                    model_name='svm_{}_{}s_{}c'.format(
                        fe_name,
                        num_samples if num_samples > -1 else 'all',
                        dim_red_option if dim_red_option is not None else 'all'
                    )
                )
                print('Time spend: {:.2f} s'.format(time.time() - temp))
                temp = time.time()

                # Feature extraction with sift, prediction with SVM and aggregation to obtain final class
                print('Predicting test data...')
                result = joblib.Parallel(n_jobs=N_JOBS, backend='threading')(
                    joblib.delayed(predict_function)(test_image, test_label, svm, std_scaler, pca) for
                    test_image, test_label in
                    zip(test_images_filenames, test_labels))
                print('Time spend: {:.2f} s'.format(time.time() - temp))
                temp = time.time()

                correct_class = [i[0] for i in result]
                predicted = [i[1] for i in result]
                expected = [i[2] for i in result]

                # Compute accuracy
                num_correct = np.count_nonzero(correct_class)
                accuracy = num_correct * 100.0 / len(test_images_filenames)

                # Show results and timing
                print('\nACCURACY: {:.2f}'.format(accuracy))
                print('TOTAL TIME: {:.2f} s'.format(time.time() - start))
                print('------------------------------')

                # Store it in object
                results.append([fe_name, num_samples, dim_red_option, accuracy])

                # Confusion matrix
                conf = metrics.confusion_matrix(expected, predicted, labels=svm.classes_)
                # Plot normalized confusion matrix
                # plot_confusion_matrix(conf, classes=svm.classes_, normalize=True)

                io.save_object(conf, 'conf_matrix_svm_{}_{}s_{}c'.format(
                    fe_name,
                    num_samples if num_samples > -1 else 'all',
                    dim_red_option if dim_red_option is not None else 'all'
                )
                )
