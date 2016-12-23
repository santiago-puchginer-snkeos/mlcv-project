from __future__ import print_function

import cPickle
import time

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

import mlcv.feature_extraction as feature_extraction
import mlcv.input_output as io

start = time.time()

# Read the training set
train_images_filenames, train_labels = io.load_training_set()
print('Loaded {} training images filenames with classes {}.'.format(len(train_images_filenames), set(train_labels)))

# Read the test set
test_images_filenames, test_labels = io.load_test_set()
print('Loaded {} testing images filenames with classes {}.'.format(len(test_images_filenames), set(test_labels)))

# Extract features
Train_descriptors = []
Train_label_per_descriptor = []

for i, filename in enumerate(train_images_filenames):
    if Train_label_per_descriptor.count(train_labels[i]) < 30:
        print('Reading image {}'.format(filename))
        gray = io.load_grayscale_image(filename)
        kpt, des = feature_extraction.brisk(gray)
        Train_descriptors.append(des)
        Train_label_per_descriptor.append(train_labels[i])
        print('{} extracted keypoints and descriptors'.format(len(kpt)))

# Transform everything to numpy arrays
D = Train_descriptors[0]
L = np.array([Train_label_per_descriptor[0]] * Train_descriptors[0].shape[0])

for i in range(1, len(Train_descriptors)):
    D = np.vstack((D, Train_descriptors[i]))
    L = np.hstack((L, np.array([Train_label_per_descriptor[i]] * Train_descriptors[i].shape[0])))

# Train a linear SVM classifier

stdSlr = StandardScaler().fit(D)
D_scaled = stdSlr.transform(D)
print('Training the SVM classifier...')
# clf = svm.SVC(kernel='linear', C=1)
# clf.fit(D_scaled, L)
# with open('SVMdef_BRISK.pickle', 'w') as f:
#     cPickle.dump(clf, f)
with open('SVMdef_BRISK.pickle', 'r') as f:
    clf = cPickle.load(f)
print('Done!')

# get all the test data and predict their labels

num_test_images = 0
num_correct = 0
for i in range(len(test_images_filenames)):
    filename = test_images_filenames[i]
    filename = "../." + filename
    ima = cv2.imread(filename)
    gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
    kpt, des = feature_extraction.brisk(gray)

    predictions = clf.predict(stdSlr.transform(des))
    values, counts = np.unique(predictions, return_counts=True)
    predictedclass = values[np.argmax(counts)]
    print('image ' + filename + ' was from class ' + test_labels[i] + ' and was predicted ' + predictedclass)
    num_test_images += 1
    if predictedclass == test_labels[i]:
        num_correct += 1

print('Final accuracy: ' + str(num_correct * 100.0 / num_test_images))

end = time.time()
print('Done in ' + str(end - start) + ' secs.')

# 38.78% in 797 secs.
