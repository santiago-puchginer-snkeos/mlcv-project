from keras.applications.vgg16 import VGG16

import numpy as np
import matplotlib.pyplot as plt

# load VGG model
base_model = VGG16(weights='imagenet')

weights = base_model.get_layer('block1_conv1').get_weights()

#get weights, discard bias
weights = weights[0]

#sum the 3 channels
weights = np.sum(weights,axis=2)


plt.figure(figsize=(8,8))

for i in range(0,64):
    w = weights[:,:,i]
    ax1 = plt.subplot(8,8,i+1)
    ax1.set_axis_off()

    plt.imshow(w, interpolation='nearest')

plt.subplots_adjust(wspace=0, hspace=0)


weights = weights.reshape(24,24)

plt.figure()
plt.imshow(weights, interpolation='nearest')



# load VGG model
base_model = VGG16(weights=None)

weights_random = base_model.get_layer('block1_conv1').get_weights()

#get weights, discard bias
weights_random = weights_random[0]

#sum the 3 channels
weights_random = np.sum(weights_random,axis=2)


plt.figure(figsize=(8,8))

for i in range(0,64):
    w = weights_random[:,:,i]
    ax1 = plt.subplot(8,8,i+1)
    ax1.set_axis_off()

    plt.imshow(w, interpolation='nearest')

plt.subplots_adjust(wspace=0, hspace=0)

weights_random = weights_random.reshape(24,24)

plt.figure()
plt.imshow(weights_random, interpolation='nearest')

plt.show()