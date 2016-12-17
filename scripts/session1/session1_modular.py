from __future__ import print_function, division

import time

import numpy as np

import mlcv.io as io
import mlcv.feature_extraction as feature_extraction
import mlcv.classification as classification

start = time.time()

# Read the training set
train_images_filenames, train_labels = io.load_training_set()
print('Loaded {} train images.'.format(len(train_images_filenames)))

# Feature extraction with sift
print('Obtaining sift features...')
D, L, _ = feature_extraction.seq_sift(train_images_filenames, train_labels, num_samples_class=30)
print('Time spend: {:.2f} s'.format(time.time() - start))
temp = time.time()

# Train Linear SVM classifier
print('Training the SVM classifier...')
lin_svm, std_scaler = classification.train_linear_svm(D, L, model_name='default_svm_sift')
print('Time spend: {:.2f} s'.format(time.time() - temp))
temp = time.time()

# Read the test set
test_images_filenames, test_labels = io.load_test_set()
print('Loaded {} test images.'.format(len(test_images_filenames)))

# Feature extraction with sift
print('Obtaining sift features...')
D, L, I = feature_extraction.seq_sift(test_images_filenames, test_labels)
print('Time spend: {:.2f} s'.format(time.time() - temp))
temp = time.time()

# Predict labels for all sift descriptors
print('Predicting labels for all descriptors...')
predicted_labels = classification.predict_linear_svm(D, lin_svm, std_scaler)
print('Time spend: {:.2f} s'.format(time.time() - temp))
temp = time.time()

# Aggregate predictions
print('Aggregating predictions of descriptors to obtain a single label...')
num_correct = 0
for i in range(len(test_images_filenames)):
    predictions_image = predicted_labels[I == i]
    values, counts = np.unique(predictions_image, return_counts=True)
    predicted_class = values[np.argmax(counts)]
    if predicted_class == test_labels[i]:
        num_correct += 1
print('Time spend: {:.2f} s'.format(time.time() - temp))
temp = time.time()

# Compute accuracy
accuracy = num_correct * 100.0 / len(test_images_filenames)

# Show results and timing
print('\nACCURACY: {}'.format(accuracy))
print('\nTOTAL TIME: {} s'.format(time.time() - start))

