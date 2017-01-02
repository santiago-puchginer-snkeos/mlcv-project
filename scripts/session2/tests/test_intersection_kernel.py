from __future__ import print_function, division

import time

import numpy as np

from mlcv.kernels import intersection_kernel

""" TEST CORRECTENESS """

# Array X
X = np.array([
    [2, 2, 2, 2],
    [4, 0, 4, 0],
    [1, 1, 1, 5]
])
print('X array:\n {}'.format(X))

# Array Y
Y = np.array([
    [1, 5, 6, 4],
    [3, 2, 5, 8]
])
print('Y array:\n {}'.format(Y))

# Expected result
exp_res = np.array([
    [7, 8],
    [5, 7],
    [7, 8]
])

# Computed result
result = intersection_kernel(X, Y)

# Comparison
print('Expected result dimensions: {}, {}'.format(X.shape[0], Y.shape[0]))
print('Computed result dimensions: {}, {}'.format(*result.shape))

print('Expected result array:\n {}'.format(exp_res))
print('Computed result array:\n {}'.format(result))

""" TEST EFFICIENCY """

# Computationally efficient version
X = np.random.randn(1500, 64)
Y = np.random.randn(1500, 64)

start = time.time()
gram = intersection_kernel(X, Y)
print('Elapsed time for {} x {} arrays: {:.2f}'.format(X.shape[0], X.shape[1], time.time() - start))

# Memory efficient version
X = np.random.randn(1500, 2**12)
Y = np.random.randn(1500, 2**12)

start = time.time()
gram_2 = intersection_kernel(X, Y)
print('Elapsed time for {} x {} arrays: {:.2f}'.format(X.shape[0], X.shape[1], time.time() - start))
