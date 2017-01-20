from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras import backend as K
from keras.utils.visualize_util import plot

import numpy as np
#import matplotlib.pyplot as plt

# load VGG model
base_model = VGG16(weights='imagenet')

# visualize topology in an image
plot(base_model, to_file='modelVGG16.png', show_shapes=True, show_layer_names=True)

# read and process image
img_path = '/data/MIT/test/coast/art1130.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# crop the model up to a certain layer
model = Model(input=base_model.input, output=base_model.get_layer('block5_conv2').output)


# get the features from images
features = model.predict(x)

if K.image_dim_ordering() == 'th':
    # theano and thensorflow deal with tensor in different order
    pass

weights = base_model.get_layer('block1_conv1').get_weights()

