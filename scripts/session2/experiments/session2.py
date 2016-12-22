import cPickle
import time

import cv2
import numpy as np
from sklearn import cluster
from sklearn import svm
from sklearn.preprocessing import StandardScaler

start = time.time()

# read the train and test files

train_images_filenames = cPickle.load(open('./dataset/train_images_filenames.dat', 'r'))
test_images_filenames = cPickle.load(open('./dataset/test_images_filenames.dat', 'r'))
train_labels = cPickle.load(open('./dataset/train_labels.dat', 'r'))
test_labels = cPickle.load(open('./dataset/test_labels.dat', 'r'))

print 'Loaded ' + str(len(train_images_filenames)) + ' training images filenames with classes ', set(train_labels)
print 'Loaded ' + str(len(test_images_filenames)) + ' testing images filenames with classes ', set(test_labels)

# create the SIFT detector object

SIFTdetector = cv2.SIFT(nfeatures=100)

# read the just 30 train images per class
# extract SIFT keypoints and descriptors
# store descriptors in a python list of numpy arrays

Train_descriptors = []
Train_label_per_descriptor = []

for i in range(len(train_images_filenames)):
    filename = train_images_filenames[i]
    print 'Reading image ' + filename
    ima = cv2.imread(filename)
    gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
    kpt, des = SIFTdetector.detectAndCompute(gray, None)
    Train_descriptors.append(des)
    Train_label_per_descriptor.append(train_labels[i])
    print str(len(kpt)) + ' extracted keypoints and descriptors'

# Transform everything to numpy arrays
size_descriptors = Train_descriptors[0].shape[1]
D = np.zeros((np.sum([len(p) for p in Train_descriptors]), size_descriptors), dtype=np.uint8)
startingpoint = 0
for i in range(len(Train_descriptors)):
    D[startingpoint:startingpoint + len(Train_descriptors[i])] = Train_descriptors[i]
    startingpoint += len(Train_descriptors[i])

k = 512

print 'Computing kmeans with ' + str(k) + ' centroids'
init = time.time()
codebook = cluster.MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=k * 20, compute_labels=False,
                                   reassignment_ratio=10 ** -4)
codebook.fit(D)
cPickle.dump(codebook, open("std_codebook.dat", "wb"))
end = time.time()
print 'Done in ' + str(end - init) + ' secs.'

init = time.time()
visual_words = np.zeros((len(Train_descriptors), k), dtype=np.float32)
for i in xrange(len(Train_descriptors)):
    words = codebook.predict(Train_descriptors[i])
    visual_words[i, :] = np.bincount(words, minlength=k)

end = time.time()
print 'Done in ' + str(end - init) + ' secs.'

# Train a linear SVM classifier

stdSlr = StandardScaler().fit(visual_words)
D_scaled = stdSlr.transform(visual_words)
print 'Training the SVM classifier...'
clf = svm.SVC(kernel='linear', C=1).fit(D_scaled, train_labels)
print 'Done!'

# get all the test data and predict their labels
visual_words_test = np.zeros((len(test_images_filenames), k), dtype=np.float32)
for i in range(len(test_images_filenames)):
    filename = test_images_filenames[i]
    print 'Reading image ' + filename
    ima = cv2.imread(filename)
    gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
    kpt, des = SIFTdetector.detectAndCompute(gray, None)
    words = codebook.predict(des)
    visual_words_test[i, :] = np.bincount(words, minlength=k)

accuracy = 100 * clf.score(stdSlr.transform(visual_words_test), test_labels)

print 'Final accuracy: ' + str(accuracy)

end = time.time()
print 'Done in ' + str(end - start) + ' secs.'

## 49.56% in 285 secs.
