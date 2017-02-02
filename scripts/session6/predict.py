from __future__ import print_function, division

from keras.applications.vgg16 import VGG16
from keras.layers import Dense, MaxPooling2D, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix

import mlcv.plotting as plotting
import numpy as np
from mlcv.cnn import preprocess_input
import mlcv.input_output as io
import mlcv.cnn as cnn

""" WEIGHTS """
weigths_file = '/home/master/anna/weights/cnn_baseline_awgn_nobatchnorm_sigma_0.05.hdf5'

""" CONSTANTS """
train_data_dir = './dataset/MIT_split/train'
val_data_dir = './dataset/MIT_split/validation'
test_data_dir = './dataset/MIT_split/test'
img_width = 128
img_height = 128
samples_epoch = 20000
val_samples_epoch = 800
test_samples = 800
number_of_epoch = 50

# Hyperparameters
regularization = 0.0001
batch_size = 100
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=10 ** (-4))
dropout = 0.75
sigma = 0.05

""" TEST DATASET """
test_images, test_labels = io.load_test_set()

""" MODEL """
# Get the base pre-trained model


# Create new model, load the weigths and compile it
model = cnn.baseline_cnn_awgn_nobatchnorm(img_width, img_height, regularization=regularization, sigma=sigma)
model.load_weights(weigths_file)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Test images generator
# Data generators
datagen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True,
                             rotation_range=10,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             preprocessing_function=cnn.preprocess_input)

test_generator = datagen.flow_from_directory(test_data_dir,
                                             shuffle=False,
                                             target_size=(img_width, img_height),
                                             batch_size=batch_size,
                                             class_mode='categorical')

""" TEST """
print('\n--------------------------------')
print('EVALUATING PERFORMANCE ON TEST SET')
print('--------------------------------\n')
result = model.evaluate_generator(test_generator, val_samples=len(test_labels))
print('Loss: {:.2f} \t Accuracy: {:.2f} %'.format(result[0], result[1] * 100))

print('\n--------------------------------')
print('COMPUTING CONFUSION MATRIX')
print('--------------------------------\n')
probs = model.predict_generator(test_generator, val_samples=len(test_labels))

classes = ['Opencountry', 'coast', 'forest', 'highway', 'inside_city', 'mountain', 'street', 'tallbuilding']
index_classes = np.argmax(probs, axis=1)
predicted_class = []
for i in index_classes:
    predicted_class.append(classes[i])
predicted_class = np.array(predicted_class)
conf = confusion_matrix(test_labels, predicted_class, labels=classes)
plotting.plot_confusion_matrix(conf, classes=classes, normalize=True)
