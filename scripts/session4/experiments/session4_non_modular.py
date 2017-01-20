from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras import backend as K
from keras.utils.visualize_util import plot

import numpy as np
import matplotlib.pyplot as plt
import cPickle

import mlcv.input_output  as io
import mlcv.kernels as kernels
from libraries.yael.yael import ynumpy
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import sklearn.decomposition as decomposition
import sklearn.preprocessing as preprocessing


""" MAIN SCRIPT"""
if __name__ == '__main__':

    k = 32
    C=1
    pca_reduction = 256

    # load VGG model
    base_model = VGG16(weights='imagenet')

    # crop the model up to a certain layer
    model = Model(input=base_model.input, output=base_model.get_layer('block5_conv2').output)

    # get train and test images
    train_images_filenames = cPickle.load(open('./dataset/train_images_filenames.dat', 'r'))
    test_images_filenames = cPickle.load(open('./dataset/test_images_filenames.dat', 'r'))
    train_labels = cPickle.load(open('./dataset/train_labels.dat', 'r'))
    test_labels = cPickle.load(open('./dataset/test_labels.dat', 'r'))
    io.log('\nLoaded {} train images.'.format(len(train_images_filenames)))
    io.log('\nLoaded {} test images.'.format(len(test_images_filenames)))


    # read and process training images
    print 'Getting features from training images'
    Train_descriptors = []
    Train_label_per_descriptor = []

    for i in range(len(train_images_filenames)):

        img = image.load_img(train_images_filenames[i], target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # get the features from images
        features = model.predict(x)
        features = features[0, :, :, :]
        descriptor = features.reshape(features.shape[0]*features.shape[1], features.shape[2])

        Train_descriptors.append(descriptor)
        Train_label_per_descriptor.append(train_labels[i])

    # Put all descriptors in a numpy array to compute PCA and GMM
    size_descriptors = Train_descriptors[0].shape[1]
    Desc = np.zeros((np.sum([len(p) for p in Train_descriptors]), size_descriptors), dtype=np.uint8)
    startingpoint = 0
    for i in range(len(Train_descriptors)):
        Desc[startingpoint:startingpoint + len(Train_descriptors[i])] = Train_descriptors[i]
        startingpoint += len(Train_descriptors[i])

    print('Computing PCA')
    pca = decomposition.PCA(n_components=pca_reduction)
    pca.fit(Desc)
    Desc = np.float32(pca.transform(Desc))

    print('Computing gmm with ' + str(k) + ' centroids')
    gmm = ynumpy.gmm_learn(np.float32(Desc), k)
    io.save_object(gmm, 'gmm_NN_pca256')


    # Compute the fisher vectors of the training images
    print('Computing fisher vectors')
    fisher = np.zeros((len(Train_descriptors), k * pca_reduction * 2), dtype=np.float32)
    for i in xrange(len(Train_descriptors)):
        descriptor = Train_descriptors[i]
        descriptor = np.float32(pca.transform(descriptor))
        fisher[i, :] = ynumpy.fisher(gmm, descriptor, include=['mu', 'sigma'])
        # L2 normalization - reshape to avoid deprecation warning, checked that the result is the same
        fisher[i, :] = preprocessing.normalize(fisher[i, :].reshape(1,-1), norm='l2')


    # Train an SVM classifier
    stdSlr = StandardScaler().fit(fisher)
    D_scaled = stdSlr.transform(fisher)
    print 'Training the SVM classifier...'
    clf = svm.SVC(kernel=kernels.intersection_kernel, C=C, probability=True).fit(D_scaled, train_labels)
    io.save_object(clf, 'clf_NN_pca256')
    #clf = io.load_object('clf_NN',ignore=False)

    # get all the test data and predict their labels
    fisher_test = np.zeros((len(test_images_filenames), k * pca_reduction * 2), dtype=np.float32)
    for i in range(len(test_images_filenames)):
        img = image.load_img(test_images_filenames[i], target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # get the features from images
        features = model.predict(x)
        features = features[0, :, :, :]
        descriptor = features.reshape(features.shape[0] * features.shape[1], features.shape[2])
        # pca
        descriptor = np.float32(pca.transform(descriptor))
        # fisher vector
        fisher_test[i, :] = ynumpy.fisher(gmm, descriptor, include=['mu', 'sigma'])
        # L2 normalization
        fisher_test[i, :] = preprocessing.normalize(fisher_test[i, :].reshape(1,-1), norm='l2')

    accuracy = 100 * clf.score(stdSlr.transform(fisher_test), test_labels)

    print 'Final accuracy: ' + str(accuracy)
