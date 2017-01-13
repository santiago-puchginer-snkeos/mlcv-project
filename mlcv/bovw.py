import numpy as np
import sklearn.cluster as cluster
import mlcv.input_output as io
import mlcv.settings as settings
from yael import ynumpy


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

def create_gmm(D, codebook_name):
    k = settings.codebook_size
    if codebook_name is not None:
        # Try to load a previously trained codebook
        try:
            gmm = io.load_object(codebook_name)
        except (IOError, EOFError):
            gmm = ynumpy.gmm_learn(D, k)
            # Store the model with the provided name
            io.save_object(gmm, codebook_name)
    else:
        gmm = ynumpy.gmm_learn(D, k)

    return gmm

def visual_words(X, y, descriptors_indices, codebook, normalization=None, spatial_pyramid=False):

    prediction = codebook.predict(X)
    v_words=[]
    if spatial_pyramid==False:
        v_words = np.array([np.bincount(prediction[descriptors_indices == i], minlength=settings.codebook_size) for i in
                   range(0, descriptors_indices.max() + 1)], dtype=np.float64)
    elif spatial_pyramid==True:
        v_words = build_pyramid(prediction, descriptors_indices)

    # Normalization
    if normalization == 'l1':
        vis_words = v_words / np.linalg.norm(v_words, axis=1, keepdims=True)
    elif normalization == 'l2':
        vis_words = v_words / np.linalg.norm(v_words, axis=1, keepdims=True) ** 2
    else:
        vis_words = v_words

    labels = [y[descriptors_indices == i][0] for i in
              range(0, descriptors_indices.max() + 1)]
    return vis_words, np.array(labels)

def fisher_vectors(X, y, descriptors_indices, codebook, normalization='l1', spatial_pyramid=False):
    fv = ynumpy.fisher(codebook, X, include=['mu', 'sigma'])
    #TODO: Spatial Pyramid Option

    #Normalization
    if normalization == 'l1':
        fisher_vect = fv / np.linalg.norm(fv, keepdims=True)
    elif normalization == 'l2':
        fisher_vect = fv / np.linalg.norm(fv, keepdims=True) ** 2
    else:
        fisher_vect = fv
    labels = [y[descriptors_indices == i][0] for i in
              range(0, descriptors_indices.max() + 1)]
    #fisher_vect.reshape(len(fisher_vect), 1)
    return fisher_vect, np.array(labels)


def build_pyramid(prediction, descriptors_indices):

    levels=settings.pyramid_levels

    v_words = []

    #Build representation for each image
    for i in range(0, descriptors_indices.max() + 1):

        image_predictions = prediction[descriptors_indices == i]
        #image_keypoints = keypoints[descriptors_indices == i]

        im_representation = 0.25*np.bincount(image_predictions, minlength=settings.codebook_size)

        keypoints_shape = int(settings.get_keypoints_shape())
        image_predictions_grid = np.reshape(image_predictions,keypoints_shape)
        kp_i = keypoints_shape[0]
        kp_j = keypoints_shape[1]

        for level in range(0,len(levels)):
            num_rows = levels[level][0]
            num_cols = levels[level][1]
            step = int(settings.dense_sampling_density)

            weight = 0.25

            if level==1:
                weight = 0.5
            for i in range(0,kp_i,step):
                for j in range(0,kp_j,step):
                    hist = weight*np.array(np.bincount(image_predictions_grid[i:i+kp_i/num_rows, j:j+kp_j/num_cols].reshape(-1), minlength=k))
                    im_representation = np.hstack((im_representation,hist))


        v_words.append(im_representation)


    return np.array(v_words, dtype=np.float64)
