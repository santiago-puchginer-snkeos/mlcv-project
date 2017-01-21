from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras import backend as K
from keras.utils.visualize_util import plot

import numpy as np
import matplotlib.pyplot as plt
import cPickle
import os
import time
import argparse
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import itertools

import mlcv.input_output  as io
import mlcv.kernels as kernels
from libraries.yael.yael import ynumpy
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import sklearn.decomposition as decomposition
import sklearn.preprocessing as preprocessing

""" PARAMETER SWEEP """

codebook_size = [16, 32, 64]
pca_reduction = [0.5, 0.4, 0.3, 0.2, 0.1]
params_distribution = {
    'C': np.logspace(-5, 3, 10 ** 6)
}
n_iter = 50


def train():
    best_accuracy = 0
    best_params = {}
    cv_results = {}

    base_model = VGG16(weights='imagenet')

    # crop the model up to a certain layer
    model = Model(input=base_model.input, output=base_model.get_layer('block5_conv2').output)

    # Read the training set
    train_images_filenames = cPickle.load(open('./dataset/train_images_filenames.dat', 'r'))
    test_images_filenames = cPickle.load(open('./dataset/test_images_filenames.dat', 'r'))
    train_labels = cPickle.load(open('./dataset/train_labels.dat', 'r'))
    test_labels = cPickle.load(open('./dataset/test_labels.dat', 'r'))
    io.log('\nLoaded {} train images.'.format(len(train_images_filenames)))
    io.log('\nLoaded {} test images.'.format(len(test_images_filenames)))

    # read and process training images
    print 'Getting features from training images'
    start_feature = time.time()
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
        descriptor = features.reshape(features.shape[0] * features.shape[1], features.shape[2])

        Train_descriptors.append(descriptor)
        Train_label_per_descriptor.append(train_labels[i])

    # Put all descriptors in a numpy array to compute PCA and GMM
    size_descriptors = Train_descriptors[0].shape[1]
    Desc = np.zeros((np.sum([len(p) for p in Train_descriptors]), size_descriptors), dtype=np.uint8)
    startingpoint = 0
    for i in range(len(Train_descriptors)):
        Desc[startingpoint:startingpoint + len(Train_descriptors[i])] = Train_descriptors[i]
        startingpoint += len(Train_descriptors[i])
    feature_time = time.time() - start_feature
    io.log('Elapsed time: {:.2f} s'.format(feature_time))

    for dim_red in pca_reduction:
        io.log('Applying PCA ... ' )
        start_pca = time.time()
        reduction = np.int(dim_red*Desc.shape[1])
        pca = decomposition.PCA(n_components=reduction)
        pca.fit(Desc)
        Desc_pca = np.float32(pca.transform(Desc))
        pca_time = time.time() - start_pca
        io.log('Elapsed time: {:.2f} s'.format(pca_time))
        for k in codebook_size:
            io.log('Creating GMM model (k = {})'.format(k))
            start_gmm = time.time()
            gmm = ynumpy.gmm_learn(np.float32(Desc_pca), k)
            io.save_object(gmm, 'gmm_NN_pca_{}_k_{}'.format(reduction, k))
            gmm_time = time.time() - start_gmm
            io.log('Elapsed time: {:.2f} s'.format(gmm_time))

            io.log('Getting Fisher vectors from training set...')
            start_fisher = time.time()
            fisher = np.zeros((len(Train_descriptors), k * reduction * 2), dtype=np.float32)
            for i in xrange(len(Train_descriptors)):
                descriptor = Train_descriptors[i]
                descriptor = np.float32(pca.transform(descriptor))
                fisher[i, :] = ynumpy.fisher(gmm, descriptor, include=['mu', 'sigma'])
                # L2 normalization - reshape to avoid deprecation warning, checked that the result is the same
                fisher[i, :] = preprocessing.normalize(fisher[i, :].reshape(1, -1), norm='l2')

            fisher_time = time.time() - start_fisher
            io.log('Elapsed time: {:.2f} s'.format(fisher_time))

            io.log('Scaling features...')
            start_scaler = time.time()
            stdSlr = StandardScaler().fit(fisher)
            D_scaled = stdSlr.transform(fisher)
            scaler_time = time.time() - start_scaler
            io.log('Elapsed time: {:.2f} s'.format(scaler_time))

            io.log('Optimizing SVM hyperparameters...')
            start_crossvalidation = time.time()
            svm = SVC(kernel='precomputed')
            random_search = RandomizedSearchCV(
                svm,
                params_distribution,
                n_iter=n_iter,
                scoring='accuracy',
                refit=False,
                cv=3,
                verbose=1
            )
            # Precompute Gram matrix
            gram = kernels.intersection_kernel(D_scaled, D_scaled)
            random_search.fit(gram, train_labels)
            crossvalidation_time = time.time() - start_crossvalidation
            io.log('Elapsed time: {:.2f} s'.format(crossvalidation_time))

            # Convert MaskedArrays to ndarrays to avoid unpickling bugs
            results = random_search.cv_results_
            results['param_C'] = results['param_C'].data

            # Appending all parameter-scores combinations
            cv_results.update({
                (k): {
                    'cv_results': results,
                    'feature_time': feature_time,
                    'pca_time': pca_time,
                    'gmm_time': gmm_time,
                    'fisher_time': fisher_time,
                    'scaler_time': scaler_time,
                    'crossvalidation_time': crossvalidation_time,
                    'total_time': feature_time + pca_time + gmm_time + fisher_time + scaler_time + crossvalidation_time
                }
            })
            io.save_object(cv_results, 'intersection_svm_CNNfeatures', ignore=True)

            # Obtaining the parameters which yielded the best accuracy
            if random_search.best_score_ > best_accuracy:
                best_accuracy = random_search.best_score_
                best_params = random_search.best_params_
                best_params.update({'k': k, 'pca': dim_red})

            io.log('-------------------------------\n')
    io.log('\nSaving best parameters...')
    io.save_object(best_params, 'best_params_intersection_svm_CNNfeatures', ignore=True)
    best_params_file = os.path.abspath('./ignore/best_params_intersection_svm_CNNfeatures.pickle')
    io.log('Saved at {}'.format(best_params_file))

    io.log('\nSaving all cross-validation values...')
    io.save_object(cv_results, 'intersection_svm_CNNfeatures', ignore=True)
    cv_results_file = os.path.abspath('./ignore/intersection_svm_CNNfeatures.pickle')
    io.log('Saved at {}'.format(cv_results_file))

    io.log('\nBEST PARAMS')
    io.log('k={}, dim_red={}, C={} --> accuracy: {:.3f}'.format(
        best_params['k'],
        best_params['pca'],
        best_params['C'],
        best_accuracy
    ))

def plot_curve():
    io.log('Loading cross-validation values...')
    cv_values = io.load_object('intersection_svm_CNNfeatures', ignore=True)

    io.log('Loading best parameters...')
    best_params = io.load_object('best_params_intersection_svm_CNNfeatures1', ignore=True)

    io.log('Plotting...')
    colors = itertools.cycle(
        ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'darkolivegreen', 'darkviolet', 'black']
    )


    # Subplot parameters
    plt.figure(facecolor='white')
    num_subplots = len(codebook_size)
    num_columns = 1
    num_rows = np.ceil(num_subplots / num_columns)

    # All subplots
    for ind, k in enumerate(codebook_size):
        # Search dictionary
        val = cv_values[(k)]
        results = val['cv_results']
        feature_time = val['feature_time']
        pca_time = val['pca_time']
        gmm_time = val['gmm_time']
        fisher_time = val['fisher_time']
        scaler_time = val['scaler_time']
        crossvalidation_time = val['crossvalidation_time']
        total_time = val['total_time']

        # Plot
        x = results['param_C']
        y = results['mean_test_score']
        e = results['std_test_score']
        sorted_indices = x.argsort()
        x_sorted = np.asarray(x[sorted_indices], dtype=np.float64)
        y_sorted = np.asarray(y[sorted_indices], dtype=np.float64)
        e_sorted = np.asarray(e[sorted_indices], dtype=np.float64)
        color = colors.next()
        ax = plt.subplot(num_rows, num_columns, ind + 1)
        ax.set_xscale("log")
        ax.set_ylim((0.7, 0.9))
        ax.errorbar(x_sorted, y_sorted, e_sorted, linestyle='--', lw=2, marker='x', color=color)
        ax.set_title('{} Gaussians in GMM'.format(k))
        ax.set_xlabel('C')
        ax.set_ylabel('Accuracy')

        # Print information
        io.log('CODEBOOK {} '.format(k))
        io.log('-------------')
        io.log('Mean accuracy: {}'.format(y.max()))
        io.log('Std accuracy: {}'.format(e[np.argmax(y)]))
        io.log('C: {}'.format(x[np.argmax(y)]))
        io.log()
        io.log('Timing')
        io.log('\tSIFT time: {:.2f} s'.format(feature_time))
        io.log('\tPCA time: {:.2f} s'.format(pca_time))
        io.log('\tGMM time: {:.2f} s'.format(gmm_time))
        io.log('\tFisher time: {:.2f} s'.format(fisher_time))
        io.log('\tScaler time: {:.2f} s'.format(scaler_time))
        io.log('\tCV time: {:.2f} s'.format(crossvalidation_time))
        io.log('\t_________________________')
        io.log('\tTOTAL TIME: {:.2f} s'.format(total_time))
        io.log()
    plt.tight_layout()
    plt.show()
    plt.close()

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