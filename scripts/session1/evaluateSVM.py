from __future__ import print_function, division

import time

import numpy as np
import cPickle
import mlcv.input_output as io
import mlcv.feature_extraction as feature_extraction
import mlcv.classification as classification
from sklearn.decomposition import PCA

if __name__ == '__main__':

    start = time.time()

    # Read the training set
    train_images_filenames, train_labels = io.load_training_set()
    print('Loaded {} train images.'.format(len(train_images_filenames)))

    # Feature extraction with sift
    print('Obtaining sift features...')
    D, L, _ = feature_extraction.seq_sift(train_images_filenames, train_labels, num_samples_class=30)
    print('Time spend: {:.2f} s'.format(time.time() - start))
    temp = time.time()

    # Create the PCA transform
    pca = PCA(n_components=23)
    pca.fit(D)
    # Transform the training data
    D_pca = pca.transform(D)


    # Read the test set
    test_images_filenames, test_labels = io.load_test_set()
    print('Loaded {} test images.'.format(len(test_images_filenames)))

    # Feature extraction with sift
    print('Obtaining sift features...')
    D_t, L_t, I_t = feature_extraction.seq_sift(test_images_filenames, test_labels)
    print('Time spend: {:.2f} s'.format(time.time() - temp))
    svm = time.time()

    # SWEEP

    kernel = 'linear'  # {'linear','poly','rbf','sigmoid'}
    sweep_mode = 'cost'  # {'cost','params'}
    Accuracy = []
    Time = []

    if sweep_mode == 'cost':
        Cost = []
        C = np.linspace(1, 10, 2)
        sw1 = C
        sw2=[1]
    elif sweep_mode == 'params':
        D = []
        Gamma = []
        R = []
        gamma = np.linspace(0.0, 1.0, 10)
        d = np.linspace(1, 10, 10)
        r = np.linspace(1, 5, 5)
        if kernel=='poly':
            sw1=d
            sw2=r
        elif kernel=='rbf':
            sw1=gamma
            sw2=[1]
        elif kernel == 'sigmoid':
            r = np.linspace(-2, 2, 10)
            sw1 = gamma
            sw2 = r


    # Train Linear SVM classifier
    print('Training the SVM classifier...')
    print('Sweeping ' +sweep_mode+ ' for kernel type ' +kernel+'...')
    for p1 in sw1:
        for p2 in sw2:
            print('p1 value '+str(p1)+' and p2 value '+str(p2))

            if sweep_mode == 'cost':
                if kernel == 'linear':
                    svm, std_scaler = classification.train_linear_svm(D_pca, L, p1)
                elif kernel == 'poly':
                    svm, std_scaler = classification.train_poly_svm(D_pca, L, p1)
                elif kernel == 'rbf':
                    svm, std_scaler = classification.train_rbf_svm(D_pca, L, p1)
                elif kernel == 'sigmoid':
                    svm, std_scaler = classification.train_sigmoid_svm(D_pca, L, p1)
                print('Time spend: {:.2f} s'.format(time.time() - temp))
                temp = time.time()
            elif sweep_mode == 'params':
                if kernel == 'poly':
                    svm, std_scaler = classification.train_poly_svm(D_pca, L, degree=p1,coef0=p2)
                elif kernel == 'rbf':
                    svm, std_scaler = classification.train_rbf_svm(D_pca, L, gamma=p1)
                elif kernel == 'sigmoid':
                    svm, std_scaler = classification.train_sigmoid_svm(D_pca, L, gamma=p1,coef0=p2)
                print('Time spend: {:.2f} s'.format(time.time() - temp))
                temp = time.time()

            # Predict labels for all sift descriptors
            print('Predicting labels for all descriptors...')
            predicted_labels = classification.predict_svm(D_t, svm, std_scaler)
            print('Time spend: {:.2f} s'.format(time.time() - temp))
            temp = time.time()

            # Aggregate predictions
            print('Aggregating predictions of descriptors to obtain a single label...')
            num_correct = 0
            for i in range(len(test_images_filenames)):
                predictions_image = predicted_labels[I_t == i]
                values, counts = np.unique(predictions_image, return_counts=True)
                predicted_class = values[np.argmax(counts)]
                if predicted_class == test_labels[i]:
                    num_correct += 1
            print('Time spend: {:.2f} s'.format(time.time() - temp))
            temp = time.time()

            # Compute results

            Accuracy.append((num_correct * 100.0 / len(test_images_filenames)))
            Time.append((temp - start))
            if sweep_mode == 'cost':
                Cost.append(p1)
            elif sweep_mode == 'params':
                if kernel=='poly':
                    D.append(p1)
                    R.append(p2)
                elif kernel == 'rbf':
                    Gamma.append(p1)
                elif kernel == 'sigmoid':
                    Gamma.append(p1)
                    R.append(p2)

# Save the results
if sweep_mode == 'cost':
    results = [Cost, Accuracy, Time]
elif sweep_mode == 'params':
    results = [Gamma,D,R, Accuracy, Time]

file = open('ResultsSVM_'+kernel+'_'+sweep_mode+'.pickle', 'wb')
cPickle.dump(results, file)
file.close()
