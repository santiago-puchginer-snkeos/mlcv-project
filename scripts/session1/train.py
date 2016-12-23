from __future__ import print_function, division

import time

import mlcv.classification as classification
import mlcv.feature_extraction as feature_extraction
import mlcv.input_output as io

from scripts import SESSION1

if __name__ == '__main__':
    start = time.time()

    # Read the training set
    train_images_filenames, train_labels = io.load_training_set()
    print('Loaded {} train images.'.format(len(train_images_filenames)))

    # Feature extraction with sift
    print('Obtaining sift features...')
    D, L, _ = feature_extraction.parallel_sift(train_images_filenames, train_labels)
    print('Time spend: {:.2f} s'.format(time.time() - start))
    temp = time.time()

    # Train Linear SVM classifier
    print('Training the SVM classifier...')
    svm, std_scaler, pca = classification.train_rbf_svm(
        D,
        L,
        model_name=SESSION1['model'],
        save_scaler=SESSION1['scaler'],
        save_pca=SESSION1['pca']
    )
    print('Time spend: {:.2f} s'.format(time.time() - temp))
    temp = time.time()
    print('\nTOTAL TRAINING TIME: {:.2f} s'.format(time.time() - start))
