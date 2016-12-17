import cPickle
import time
# from __future__ import print_function
import cv2
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# read the train and test files

train_images_filenames = cPickle.load(open('./dataset/train_images_filenames.dat', 'r'))
test_images_filenames = cPickle.load(open('./dataset/test_images_filenames.dat', 'r'))
train_labels = cPickle.load(open('./dataset/train_labels.dat', 'r'))
test_labels = cPickle.load(open('./dataset/test_labels.dat', 'r'))

print('Loaded ' + str(len(train_images_filenames)) + ' training images filenames with classes ', set(train_labels))
print('Loaded ' + str(len(test_images_filenames)) + ' testing images filenames with classes ', set(test_labels))

# create the SIFT detector object
SIFT_detector = cv2.SIFT(nfeatures=100)

# read the just 30 train images per class
# extract SIFT keypoints and descriptors
# store descriptors in a python list of numpy arrays

Train_descriptors = []
Train_label_per_descriptor = []
Accuracy = []
Time = []
Cost = []

for i in range(len(train_images_filenames)):
    filename = train_images_filenames[i]
    if Train_label_per_descriptor.count(train_labels[i]) < 30:
        print('Reading image ' + filename)
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt, des = SIFT_detector.detectAndCompute(gray, None)
        Train_descriptors.append(des)
        Train_label_per_descriptor.append(train_labels[i])
        print(str(len(kpt)) + ' extracted keypoints and descriptors')

# Transform everything to numpy arrays

D = Train_descriptors[0]
L = np.array([Train_label_per_descriptor[0]] * Train_descriptors[0].shape[0])

for i in range(1, len(Train_descriptors)):
    D = np.vstack((D, Train_descriptors[i]))
    L = np.hstack((L, np.array([Train_label_per_descriptor[i]] * Train_descriptors[i].shape[0])))

# Create the PCA transform
pca = PCA(n_components=23)
pca.fit(D)
# Transform the training data
D_pca = pca.transform(D)

# Create an scaler
stdSlr = StandardScaler().fit(D_pca)
# Scale training data
D_scaled = stdSlr.transform(D_pca)

for c in range(1, 10):
    # Transform the data to different dimensions with PCA and train a linear SVM classifier for each value
    print('For value c: ' + str(c))
    clf = svm.SVC(kernel='sigmoid', C=c, coef0=-0.8)
    clf.fit(D_scaled, L)

    # get all the test data and predict their labels
    start = time.time()
    num_test_images = 0
    num_correct = 0
    for i in range(len(test_images_filenames)):
        filename = test_images_filenames[i]
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt, des = SIFT_detector.detectAndCompute(gray, None)
        des_pca = pca.transform(des)
        predictions = clf.predict(stdSlr.transform(des_pca))
        values, counts = np.unique(predictions, return_counts=True)
        predictedclass = values[np.argmax(counts)]
        num_test_images += 1
        if predictedclass == test_labels[i]:
            num_correct += 1
    end = time.time()

    print('Final accuracy: ' + str(num_correct * 100.0 / num_test_images))

    # Save the accuracy and time spent in the classification for each dimension
    Accuracy.append((num_correct * 100.0 / num_test_images))
    Cost.append(c)

# Save the results
results = [Cost, Accuracy, Time]
file = open('ResultsSVM_sigmoid_cost.pickle', 'wb')
cPickle.dump(results, file)
file.close()
