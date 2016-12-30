from __future__ import print_function, division

from mlcv.kernels import intersection_kernel

import numpy as np

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
