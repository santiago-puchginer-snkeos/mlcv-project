import cv2
import numpy as np
import cPickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import cluster
from yael import ynumpy

start = time.time()

# read the train and test files

train_images_filenames = cPickle.load(open('./dataset/train_images_filenames.dat','r'))
test_images_filenames = cPickle.load(open('./dataset/test_images_filenames.dat','r'))
train_labels = cPickle.load(open('./dataset/train_labels.dat','r'))
test_labels = cPickle.load(open('./dataset/test_labels.dat','r'))

print 'Loaded '+str(len(train_images_filenames))+' training images filenames with classes ',set(train_labels)
print 'Loaded '+str(len(test_images_filenames))+' testing images filenames with classes ',set(test_labels)

# create the SIFT detector object

SIFTdetector = cv2.SIFT(nfeatures=100)

# extract SIFT keypoints and descriptors
# store descriptors in a python list of numpy arrays

Train_descriptors = []
Train_label_per_descriptor = []

for i in range(len(train_images_filenames)):
	filename=train_images_filenames[i]
	print 'Reading image '+filename
	ima=cv2.imread(filename)
	gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
	kpt,des=SIFTdetector.detectAndCompute(gray,None)
	Train_descriptors.append(des)
	Train_label_per_descriptor.append(train_labels[i])
	print str(len(kpt))+' extracted keypoints and descriptors'

# Transform everything to numpy arrays
size_descriptors=Train_descriptors[0].shape[1]
D=np.zeros((np.sum([len(p) for p in Train_descriptors]),size_descriptors),dtype=np.uint8)
startingpoint=0
for i in range(len(Train_descriptors)):
	D[startingpoint:startingpoint+len(Train_descriptors[i])]=Train_descriptors[i]
	startingpoint+=len(Train_descriptors[i])


k = 32

print 'Computing gmm with '+str(k)+' centroids'
init=time.time()
gmm = ynumpy.gmm_learn(np.float32(D), k)
end=time.time()
print 'Done in '+str(end-init)+' secs.'



init=time.time()
fisher=np.zeros((len(Train_descriptors),k*128*2),dtype=np.float32)
for i in xrange(len(Train_descriptors)):
	fisher[i,:]= ynumpy.fisher(gmm, Train_descriptors[i], include = ['mu','sigma'])


end=time.time()
print 'Done in '+str(end-init)+' secs.'


# Train a linear SVM classifier

stdSlr = StandardScaler().fit(fisher)
D_scaled = stdSlr.transform(fisher)
print 'Training the SVM classifier...'
clf = svm.SVC(kernel='linear', C=1).fit(D_scaled, train_labels)
print 'Done!'

# get all the test data and predict their labels
fisher_test=np.zeros((len(test_images_filenames),k*128*2),dtype=np.float32)
for i in range(len(test_images_filenames)):
	filename=test_images_filenames[i]
	print 'Reading image '+filename
	ima=cv2.imread(filename)
	gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
	kpt,des=SIFTdetector.detectAndCompute(gray,None)
	fisher_test[i,:]=ynumpy.fisher(gmm, des, include = ['mu','sigma'])


accuracy = 100*clf.score(stdSlr.transform(fisher_test), test_labels)

print 'Final accuracy: ' + str(accuracy)

end=time.time()
print 'Done in '+str(end-start)+' secs.'

## 61.71% in 251 secs.