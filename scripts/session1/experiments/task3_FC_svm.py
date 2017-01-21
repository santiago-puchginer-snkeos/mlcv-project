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
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)

    # get train and test images
    train_images_filenames = cPickle.load(open('./dataset/train_images_filenames.dat', 'r'))
    test_images_filenames = cPickle.load(open('./dataset/test_images_filenames.dat', 'r'))
    train_labels = cPickle.load(open('./dataset/train_labels.dat', 'r'))
    test_labels = cPickle.load(open('./dataset/test_labels.dat', 'r'))
    io.log('\nLoaded {} train images.'.format(len(train_images_filenames)))

    # read and process training images
    print 'Getting features from training images'
    first=1
    for i in range(len(train_images_filenames)):
        img = image.load_img(train_images_filenames[i], target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # get the features from images
        features = model.predict(x)
        features = features[0, :]
        if first == 1:
            Desc = features
            first = 0
        else:
            Desc = np.vstack((Desc, features))

    io.save_object(Desc, 'train_descriptors')

    print('Computing PCA')
    #pca = decomposition.PCA(n_components=pca_reduction)
    #pca.fit(Desc)
    #Desc = np.float32(pca.transform(Desc))

    # Train a linear SVM classifier
    stdSlr = StandardScaler().fit(Desc)
    D_scaled = stdSlr.transform(Desc)
    print 'Training the SVM classifier...'
    clf = svm.SVC(kernel=kernels.intersection_kernel, C=C, probability=True).fit(D_scaled, train_labels)
    io.save_object(clf, 'clf_T3_pca256')
    #clf = io.load_object('clf_T3_pca256',ignore=False)

    # get all the test data and predict their labels
    features_test = np.zeros((len(test_images_filenames),  model.output_shape[1]), dtype=np.float32)
    for i in range(len(test_images_filenames)):
        img = image.load_img(test_images_filenames[i], target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # get the features from images
        descriptor = model.predict(x)
        features_test[i, :] = descriptor[0, :]
    # pca
    #features_test = np.float32(pca.transform(features_test))


    accuracy = 100 * clf.score(stdSlr.transform(features_test), test_labels)

    print 'Final accuracy: ' + str(accuracy)
