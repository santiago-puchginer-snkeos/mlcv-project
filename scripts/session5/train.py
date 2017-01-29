from __future__ import print_function, division

import time

import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers import Dense, MaxPooling2D, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

from mlcv.cnn import preprocess_input

""" CONSTANTS """
train_data_dir = './dataset/MIT_split/train'
val_data_dir = './dataset/MIT_split/validation'
test_data_dir = './dataset/MIT_split/test'
img_width = 224
img_height = 224
samples_epoch = 4000
val_samples_epoch = 400
test_samples = 800
number_of_epoch_fc = 100
number_of_epoch_full = 100

# Hyperparameters
dropout = 0.5
regularization = 0.01
batch_size = 20
lr = 1e-5
optimizer = Adam(lr=lr)

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

# Create new model and save it
model = Model(input=base_model.input, output=x)

# Data generators
datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=True,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=True,
                             zca_whitening=False,
                             rotation_range=10,
                             width_shift_range=0.,
                             height_shift_range=0.,
                             shear_range=0.,
                             zoom_range=0.,
                             channel_shift_range=0.,
                             fill_mode='nearest',
                             cval=0.,
                             horizontal_flip=True,
                             vertical_flip=False,
                             rescale=None,
                             preprocessing_function=preprocess_input)

train_generator = datagen.flow_from_directory(train_data_dir,
                                              shuffle=True,
                                              target_size=(img_width, img_height),
                                              batch_size=batch_size,
                                              class_mode='categorical')

test_generator = datagen.flow_from_directory(test_data_dir,
                                             shuffle=True,
                                             target_size=(img_width, img_height),
                                             batch_size=batch_size,
                                             class_mode='categorical')

validation_generator = datagen.flow_from_directory(val_data_dir,
                                                   shuffle=True,
                                                   target_size=(img_width, img_height),
                                                   batch_size=batch_size,
                                                   class_mode='categorical')

print('\n--------------------------------')
print('FULLY CONNECTED LAYERS TRAINING')
print('--------------------------------\n')
start_time = time.time()
for layer in base_model.layers:
    layer.trainable = False

print('\nLAYERS\n')
for layer in model.layers:
    print('NAME: {}\t TRAINABLE: {}'.format(layer.name, layer.trainable))

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history_fc = model.fit_generator(train_generator,
                                 samples_per_epoch=samples_epoch,
                                 nb_epoch=number_of_epoch_fc,
                                 validation_data=validation_generator,
                                 nb_val_samples=val_samples_epoch,
                                 callbacks=[
                                     ModelCheckpoint('./weights/final_system_fc.hdf5',
                                                     save_best_only=True,
                                                     save_weights_only=True),
                                     TensorBoard(log_dir='./tf_logs/final_system_fc'),
                                     EarlyStopping(monitor='val_loss', patience=5)
                                 ])
print('Total training time: {:.2f} s'.format(time.time() - start_time))

print('\n--------------------------------')
print('FULL NETWORK TRAINING')
print('--------------------------------\n')
start_time = time.time()
for layer in base_model.layers:
    layer.trainable = True

print('\nLAYERS\n')
for layer in model.layers:
    print('NAME: {}\t TRAINABLE: {}'.format(layer.name, layer.trainable))

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history_full = model.fit_generator(train_generator,
                                   samples_per_epoch=samples_epoch,
                                   nb_epoch=number_of_epoch_full,
                                   validation_data=validation_generator,
                                   nb_val_samples=val_samples_epoch,
                                   callbacks=[
                                       ModelCheckpoint('./weights/final_system_full.hdf5',
                                                       save_best_only=True,
                                                       save_weights_only=True),
                                       TensorBoard(log_dir='./tf_logs/final_system_full'),
                                       EarlyStopping(monitor='val_loss', patience=5)
                                   ]
                                   )
print('Total training time: {:.2f} s'.format(time.time() - start_time))

print('\n--------------------------------')
print('EVALUATING PERFORMANCE ON TEST SET')
print('--------------------------------\n')
result = model.evaluate_generator(test_generator, val_samples=test_samples)
print('Loss: {:.2f} \t Accuracy: {:.2f} %'.format(result[0], result[1] * 100))

print('\n--------------------------------')
print('STORING LOSS AND ACCURACY PLOTS')
print('--------------------------------\n')
plt.plot(history_fc.history['acc'])
plt.plot(history_fc.history['val_acc'])
plt.title('Model accuracy (only FC layers training)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim((0, 1))
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('./results/final_system_fc_accuracy.jpg')
plt.close()

plt.plot(history_full.history['acc'])
plt.plot(history_full.history['val_acc'])
plt.title('Model accuracy (whole network training)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim((0, 1))
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('./results/final_system_full_accuracy.jpg')
plt.close()

plt.plot(history_fc.history['loss'])
plt.plot(history_fc.history['val_loss'])
plt.title('Model loss (only FC layers training)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('./results/final_system_fc_loss.jpg')
plt.close()

plt.plot(history_full.history['loss'])
plt.plot(history_full.history['val_loss'])
plt.title('Model loss (whole network training)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('./results/final_system_full_loss.jpg')
plt.close()

