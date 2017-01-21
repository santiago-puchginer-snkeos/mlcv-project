from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras import backend as K
from keras.utils.visualize_util import plot
from keras.layers import  Input
from keras.layers import  MaxPooling2D
from keras.applications.imagenet_utils import  preprocess_input, _obtain_input_shape
import argparse

import numpy as np
#import matplotlib.pyplot as plt
import cPickle
import os

import mlcv.input_output  as io
import mlcv.kernels as kernels
from libraries.yael.yael import ynumpy
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import sklearn.decomposition as decomposition
import sklearn.preprocessing as preprocessing

from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

""" PARAMETER SWEEP """

codebook_size = [16, 32, 64]
params_distribution = {
    'C': np.logspace(-3, 1, 10 ** 6)
}
n_iter = 50

def train():
    best_accuracy = 0
    best_params = {}
    cv_results = {}

    # load VGG model
    base_model = VGG16(weights='imagenet')

    # crop the model up to a certain layer
    model = Model(input=base_model.input, output=base_model.get_layer('block5_conv2').output)

    # aggregating features with max-pooling
   # inputs = Input(shape=[14, 14, 512])
   # x = MaxPooling2D((2, 2), strides=(2, 2), name='max_pooling_layer')(inputs)
   # model_agg = Model(inputs, x, name='agg_features')

    # get train and test images
    train_images_filenames = cPickle.load(open('./dataset/train_images_filenames.dat', 'r'))
    train_labels = cPickle.load(open('./dataset/train_labels.dat', 'r'))
    io.log('\nLoaded {} train images.'.format(len(train_images_filenames)))


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
        features_ = model.predict(x)
        features = features_[0, :, :, :]
        descriptor = features.reshape(features.shape[0]*features.shape[1], features.shape[2])
        # aggregate features
        #descriptor_agg=descriptor.max(axis=1)
        #descriptor_agg=np.reshape(descriptor_agg,[descriptor_agg.shape[0],1])

        Train_descriptors.append(descriptor)
        Train_label_per_descriptor.append(train_labels[i])

    # Put all descriptors in a numpy array to compute PCA and GMM
    size_descriptors = Train_descriptors[0].shape[1]
    #size_descriptors=1
    Desc = np.zeros((np.sum([len(p) for p in Train_descriptors]), size_descriptors), dtype=np.uint8)
    startingpoint = 0
    for i in range(len(Train_descriptors)):
        Desc[startingpoint:startingpoint + len(Train_descriptors[i])] = Train_descriptors[i]
        startingpoint += len(Train_descriptors[i])


    for k in codebook_size:

        print('Computing gmm with ' + str(k) + ' centroids')
        gmm = ynumpy.gmm_learn(np.float32(Desc), k)
       # io.save_object(gmm, 'gmm_NN_agg_features_max')


        # Compute the fisher vectors of the training images
        print('Computing fisher vectors')
        fisher = np.zeros((len(Train_descriptors), k * 512 * 2), dtype=np.float32)
        for i in xrange(len(Train_descriptors)):
            descriptor = Train_descriptors[i]
           # descriptor = np.float32(pca.transform(descriptor))
            aux=ynumpy.fisher(gmm, descriptor, include=['mu', 'sigma'])
            #fisher[i, :] = np.reshape(aux, [1, aux.shape[0]])
            fisher[i,:]=aux
            # L2 normalization - reshape to avoid deprecation warning, checked that the result is the same
            fisher[i, :] = preprocessing.normalize(fisher[i, :].reshape(1,-1), norm='l2')


    # CV in SVM training
        io.log('Scaling features...')
        std_scaler = StandardScaler().fit(fisher)
        vis_words = std_scaler.transform(fisher)

        io.log('Optimizing SVM hyperparameters...')
        svm = SVC(kernel='precomputed')
        random_search = RandomizedSearchCV(
            svm,
            params_distribution,
            n_iter=n_iter,
            scoring='accuracy',
            n_jobs=1,
            refit=False,
            cv=3,
            verbose=1
        )
        # Precompute Gram matrix
        gram = kernels.intersection_kernel(vis_words, vis_words)
        random_search.fit(gram, train_labels)

        # Convert MaskedArrays to ndarrays to avoid unpickling bugs
        results = random_search.cv_results_
        results['param_C'] = results['param_C'].data

        # Appending all parameter-scores combinations
        cv_results.update({
            (k): {
                'cv_results': results,
                }
        })
        io.save_object(cv_results, 'intersection_svm_CNNfeatures_aggregate', ignore=True)

        # Obtaining the parameters which yielded the best accuracy
        if random_search.best_score_ > best_accuracy:
            best_accuracy = random_search.best_score_
            best_params = random_search.best_params_
            best_params.update({'k': k})

        io.log('-------------------------------\n')
    io.log('\nSaving best parameters...')
    io.save_object(best_params, 'best_params_intersection_svm_CNNfeatures', ignore=True)
    best_params_file = os.path.abspath('./ignore/best_params_intersection_svm_CNNfeatures_aggregate.pickle')
    io.log('Saved at {}'.format(best_params_file))

    io.log('\nSaving all cross-validation values...')
    io.save_object(cv_results, 'intersection_svm_CNNfeatures', ignore=True)
    cv_results_file = os.path.abspath('./ignore/intersection_svm_CNNfeatures_aggregate.pickle')
    io.log('Saved at {}'.format(cv_results_file))

    io.log('\nBEST PARAMS')
    io.log('k={}, C={} --> accuracy: {:.3f}'.format(
        best_params['k'],
        best_params['C'],
        best_accuracy
    ))
def plot_curve():
    train()

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--type', default='train', choices=['train', 'plot'])
    args = args_parser.parse_args()
    exec_option = args.type

    if exec_option == 'train':
        train()
    elif exec_option == 'plot':
        plot_curve()
    exit(0)