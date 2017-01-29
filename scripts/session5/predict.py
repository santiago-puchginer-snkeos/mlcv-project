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

""" WEIGHTS """
weigths_file = '/home/master/santi/weights/final_system_full.hdf5'

""" CONSTANTS """
test_data_dir = './dataset/MIT_split/test'
img_width = 224
img_height = 224

# Hyperparameters
dropout = 0.5
regularization = 0.01
batch_size = 20
lr = 1e-5
optimizer = Adam(lr=lr)

""" TEST DATASET """
test_images, test_labels = io.load_test_set()

""" MODEL """
# Get the base pre-trained model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Get output from last convolutional layer in block 4
x = base_model.get_layer('block4_conv3').output
x = MaxPooling2D(pool_size=(4, 4))(x)
x = Flatten(name='flat')(x)
x = Dense(2048, activation='relu', name='fc', W_regularizer=l2(regularization))(x)
x = Dropout(dropout)(x)
x = Dense(2048, activation='relu', name='fc2', W_regularizer=l2(regularization))(x)
x = Dropout(dropout)(x)
x = Dense(8, activation='softmax', name='predictions')(x)

# Create new model, load the weigths and compile it
model = Model(input=base_model.input, output=x)
model.load_weights(weigths_file)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Test images generator
datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=True,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=True,
                             zca_whitening=False,
                             rotation_range=0,
                             width_shift_range=0.,
                             height_shift_range=0.,
                             shear_range=0.,
                             zoom_range=0.,
                             channel_shift_range=0.,
                             fill_mode='nearest',
                             cval=0.,
                             horizontal_flip=False,
                             vertical_flip=False,
                             rescale=None,
                             preprocessing_function=preprocess_input)

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
