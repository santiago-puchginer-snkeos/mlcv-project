import numpy as np


def intersection_kernel(X, Y):
    minim = np.minimum(X[:, :, None], Y[:, :, None].T)
    return np.sum(minim, axis=1)
