import mlcv.input_output as io
import numpy as np
from matplotlib import pyplot as plt


weights_bias = io.load_object('weights_first_layer',True)

weights_ = weights_bias[0]

weights = np.sum(weights_,axis=2)

weights = weights.reshape(24,24)

plt.figure()
plt.imshow(weights, interpolation='nearest')


weights_bias_random = io.load_object('weights_first_layer_random',True)

weights_random_ = weights_bias_random[0]

weights_random = np.sum(weights_random_,axis=2)

weights_random = weights_random.reshape(24,24)


plt.figure()
plt.imshow(weights_random, interpolation='nearest')
plt.show()

a=1

