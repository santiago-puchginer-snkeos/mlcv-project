import numpy as np
import sklearn.cluster as cluster
from libraries.yael.yael import ynumpy


import mlcv.input_output as io
import mlcv.settings as settings

import math



def create_codebook(X, codebook_name=None, k_means_init='random'):
    k = settings.codebook_size
    batch_size = 20 * k if X.shape[0] > 20 * k else X.shape[0] / 10
    codebook = cluster.MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=batch_size, compute_labels=False,
                                       reassignment_ratio=10 ** -4, init=k_means_init)

    if codebook_name is not None:
        # Try to load a previously trained codebook
        try:
            codebook = io.load_object(codebook_name)
        except (IOError, EOFError):
            codebook.fit(X)
            # Store the model with the provided name
            io.save_object(codebook, codebook_name)
    else:
        codebook.fit(X)

    return codebook


def create_gmm(D, codebook_name=None):
    from libraries.yael.yael import ynumpy

    k = settings.codebook_size
    if codebook_name is not None:
        # Try to load a previously trained codebook
        try:
            gmm = io.load_object(codebook_name)
        except (IOError, EOFError):
            gmm = ynumpy.gmm_learn(np.float32(D), k)
            # Store the model with the provided name
            io.save_object(gmm, codebook_name)
    else:
        gmm = ynumpy.gmm_learn(np.float32(D), k)

    return gmm


def visual_words(X, y, descriptors_indices, codebook, normalization=None, spatial_pyramid=False):
    prediction = codebook.predict(X)
    if not spatial_pyramid:
        v_words = np.array([np.bincount(prediction[descriptors_indices == i], minlength=settings.codebook_size) for i in
                            range(0, descriptors_indices.max() + 1)], dtype=np.float64)
    else:
        v_words = build_pyramid(prediction, descriptors_indices)

    # Normalization
    if normalization == 'l1':
        vis_words = v_words / np.sum(np.abs(v_words), axis=1, keepdims=True)
    elif normalization == 'l2':
        vis_words = v_words / np.linalg.norm(v_words, axis=1, keepdims=True)
    else:
        vis_words = v_words

    labels = [y[descriptors_indices == i][0] for i in
              range(0, descriptors_indices.max() + 1)]
    return vis_words, np.array(labels)


def fisher_vectors(X, y, descriptors_indices, codebook, normalization=None, spatial_pyramid=False):
    from libraries.yael.yael import ynumpy

    # Compute Fisher vector for each image (which can have multiple descriptors)
    X = np.float32(X)
    fv = np.array(
        [ynumpy.fisher(
            codebook,
            X[descriptors_indices == i],
            include=['mu', 'sigma']) for i in range(0, descriptors_indices.max() + 1)]
    )
    # TODO: Spatial Pyramid Option

    # Normalization
    if normalization == 'l1':
        fisher_vect = fv / np.sum(np.abs(fv), axis=1, keepdims=True)
    elif normalization == 'l2':
        fisher_vect = fv / np.linalg.norm(fv, keepdims=True)
    elif normalization == 'power':
        fisher_vect = np.multiply(np.sign(fv)* np.sqrt(np.absolute(fv)))
    else:
        fisher_vect = fv
    labels = [y[descriptors_indices == i][0] for i in
              range(0, descriptors_indices.max() + 1)]
    return fisher_vect, np.array(labels)


def build_pyramid(prediction, descriptors_indices):

    levels = settings.pyramid_levels
    keypoints_shape = map(int, settings.get_keypoints_shape())
    kp_i = keypoints_shape[0]
    kp_j = keypoints_shape[1]

    v_words = []

    # Build representation for each image
    for i in range(0, descriptors_indices.max() + 1):

        image_predictions = prediction[descriptors_indices == i]
        image_predictions_grid = np.reshape(image_predictions, keypoints_shape)

        im_representation = []


        for level in range(0, len(levels)):
            num_rows = levels[level][0]
            num_cols = levels[level][1]
            step_i = int(math.ceil(float(kp_i)/float(num_rows)))
            step_j = int(math.ceil(float(kp_j)/float(num_cols)))

            for i in range(0,kp_i,step_i):
                for j in range(0,kp_j,step_j):
                    hist = np.array(np.bincount(image_predictions_grid[i:i+step_i, j:j+step_j].reshape(-1), minlength=settings.codebook_size))
                    im_representation = np.hstack((im_representation,hist))

        v_words.append(im_representation)

    return np.array(v_words, dtype=np.float64)
