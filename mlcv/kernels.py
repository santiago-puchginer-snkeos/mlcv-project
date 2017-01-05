import numpy as np

MAX_NUM_ELEMS = 1e8


def intersection_kernel(X, Y):
    """
    Computes the Gram matrix between matrix X and Y using the intersection kernel.
    K(x_i, x_j) = sum_{k=0}^{n_features} min(x_ik, x_jk)

    :param X: X matrix with dimensions (n_samples_x, n_features)
    :type X: numpy.ndarray
    :param Y: Y matrix with dimensions (n_samples_y, n_features)
    :type Y: numpy.ndarray

    :return: Gram matrix with dimensions (n_samples_x, n_samples_y)
    :rtype: numpy.ndarray
    """
    n_samples_x, n_features = X.shape
    n_samples_y, _ = Y.shape

    if n_samples_x * n_samples_y * n_features < MAX_NUM_ELEMS:
        # Computationally efficient version (requires more memory)
        minim = np.minimum(X[:, :, None], Y[:, :, None].T)
        intersection = np.sum(minim, axis=1)
    else:
        # Memory efficient version (requires more computation time)
        x_slicing = int(np.floor(MAX_NUM_ELEMS / (n_samples_y * n_features)))
        intersection = np.zeros((n_samples_x, n_samples_y))

        for i in range(0, n_samples_x, x_slicing):
            minim = np.minimum(X[i:i + x_slicing, :, None], Y[:, :, None].T)
            intersection[i:i + x_slicing, :] = np.sum(minim, axis=1)

    return intersection

def pyramid_kernel(X, Y):
    codebook_size = len(X[0,:])/21
    intersection = 0
    for i in range(0,len(X[0,:]),codebook_size):
        intersection = intersection + intersection_kernel(X[:,i:i+codebook_size], Y[:,i:i+codebook_size])

    return intersection