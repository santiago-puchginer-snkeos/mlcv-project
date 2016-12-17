import cPickle
import time
import sys
sys.path.insert(0, r'../../')
import cv2
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import mlcv.feature_extraction as feature_extraction
from sklearn.decomposition import PCA

start = time.time()
#descriptor_option = sys.argv[1]
# read the train and test files

train_images_filenames = cPickle.load(open('../../dataset/train_images_filenames.dat', 'r'))
test_images_filenames = cPickle.load(open('../../dataset/test_images_filenames.dat', 'r'))
train_labels = cPickle.load(open('../../dataset/train_labels.dat', 'r'))
test_labels = cPickle.load(open('../../dataset/test_labels.dat', 'r'))

print 'Loaded ' + str(len(train_images_filenames)) + ' training images filenames with classes ', set(train_labels)
print 'Loaded ' + str(len(test_images_filenames)) + ' testing images filenames with classes ', set(test_labels)
print train_images_filenames[1]


# read the just 30 train images per class
# extract SIFT keypoints and descriptors
# store descriptors in a python list of numpy arrays

Train_descriptors = []
Train_label_per_descriptor = []
final_accuracy=[]
for sigma1 in range(10, 30, 1):
    sigma = float(sigma1)/10
    for ct1 in range(1, 9):
        ct = float(ct1)/100
        for et in range(28, 88, 10):
            for i in range(len(train_images_filenames)):
                filename = train_images_filenames[i]
                filename = "../." + filename
                if Train_label_per_descriptor.count(train_labels[i]) < 30:
                    print 'Reading image ' + filename
                    ima = cv2.imread(filename)
                    gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
                    ##kpt, des = SIFT_detector.detectAndCompute(gray, None)
                    kpt, des = feature_extraction.FREAK(gray, sigma, ct, et)
                    Train_descriptors.append(des)
                    Train_label_per_descriptor.append(train_labels[i])
                    print str(len(kpt)) + ' extracted keypoints and descriptors'

            # Transform everything to numpy arrays

            D = Train_descriptors[0]
            L = np.array([Train_label_per_descriptor[0]] * Train_descriptors[0].shape[0])

            for i in range(1, len(Train_descriptors)):
                D = np.vstack((D, Train_descriptors[i]))
                L = np.hstack((L, np.array([Train_label_per_descriptor[i]] * Train_descriptors[i].shape[0])))

            # Train a linear SVM classifier
            pca = PCA(n_components=23)
            pca.fit(D)
            # Transform the training data
            D_pca = pca.transform(D)

            # Create an scaler
            stdSlr = StandardScaler().fit(D_pca)
            # Scale training data
            D_scaled = stdSlr.transform(D_pca)
            print 'Training the SVM classifier...'
            clf = svm.SVC(kernel='linear', C=1)
            clf.fit(D_scaled, L)
            with open('SVMdef_FREAK_sigma'+str(sigma)+'_cT'+str(ct)+'_eT'+str(et)+'.pickle', 'w') as f:
                cPickle.dump(clf, f)
            #with open('SVMdef_SURF.pickle', 'r') as f:
            #    clf = cPickle.load(f)
            print 'Done!'

            # get all the test data and predict their labels

            num_test_images = 0
            num_correct = 0
            for i in range(len(test_images_filenames)):
                filename = test_images_filenames[i]
                filename = "../."+filename
                ima = cv2.imread(filename)
                gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
                kpt, des = feature_extraction.FREAK(gray, sigma, ct, et)
                des_pca = pca.transform(des)
                predictions = clf.predict(stdSlr.transform(des_pca))
                values, counts = np.unique(predictions, return_counts=True)
                predictedclass = values[np.argmax(counts)]
                print 'image ' + filename + ' was from class ' + test_labels[i] + ' and was predicted ' + predictedclass
                num_test_images += 1
                if predictedclass == test_labels[i]:
                    num_correct += 1
            accuracy = num_correct * 100.0 / num_test_images
            with open('accuracy_SVMdef_FREAK_sigma' + str(sigma) + '_cT' + str(ct) + '_eT' + str(et) + '.pickle',
                      'w') as f:
                cPickle.dump(accuracy, f)


            print 'Final accuracy: ' + str(num_correct * 100.0 / num_test_images)

            end = time.time()
            print 'Done in ' + str(end - start) + ' secs.'

            # 38.78% in 797 secs.
            tme = end - start
            with open('timeSIFT_sigma' + str(sigma) + '_cT' + str(ct) + '_eT' + str(et) + '.pickle', 'w') as f:
                cPickle.dump(tme, f)
final_accuracy = np.append(final_accuracy, accuracy)
with open('global_accuracy_FREAK.pickle', 'w') as f:
    cPickle.dump(final_accuracy, f)
