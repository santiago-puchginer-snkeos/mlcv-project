from __future__ import print_function, division

import os
import time

import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import visualize_util as keras_visualize

import mlcv.cnn as cnn
import mlcv.input_output as io

""" CONSTANTS """
train_data_dir = './dataset/MIT_split/train'
val_data_dir = './dataset/MIT_split/validation'
test_data_dir = './dataset/MIT_split/test'
img_width = 128
img_height = 128
samples_epoch = 10000
val_samples_epoch = 200
test_samples = 200
number_of_epoch = 150

# Hyperparameters
regularization = 0.1
batch_size = 150
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=10 ** (-4))
gaussian_noise = False
dropout = 0

# Results names
results_name = '{}_dropout_{}_awgn_{}'.format(
    os.path.basename(__file__).replace('.py', ''),
    dropout,
    str(gaussian_noise).lower()
)

# Create new model and save it
model = cnn.baseline_cnn_alt_dropout_fc_mod(img_width, img_height, regularization=regularization, dropout=dropout,
                                            gaussian_noise=gaussian_noise)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print('\n{:^80}\n'.format('MODEL SUMMARY'))
model.summary()
keras_visualize.plot(model, './results/{}.png'.format(results_name), show_shapes=True)

# Data generators
datagen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True,
                             rotation_range=10,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             preprocessing_function=cnn.preprocess_input)

val_datagen = ImageDataGenerator(featurewise_center=True,
                                 featurewise_std_normalization=True,
                                 preprocessing_function=cnn.preprocess_input)

# Load train dataset and fit the ImageDataGenerator
print('Fitting ImageDataGenerator to the train dataset...')
train_images, train_labels = io.load_dataset_from_directory(train_data_dir)
datagen.fit(train_images)
val_datagen.fit(train_images)

# Create the generators
train_generator = datagen.flow_from_directory(train_data_dir,
                                              shuffle=True,
                                              target_size=(img_width, img_height),
                                              batch_size=batch_size,
                                              class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(val_data_dir,
                                                       shuffle=False,
                                                       target_size=(img_width, img_height),
                                                       batch_size=batch_size,
                                                       class_mode='categorical')

print('\n--------------------------------')
print('TRAINING')
print('--------------------------------\n')
start_time = time.time()
history = model.fit_generator(train_generator,
                              samples_per_epoch=samples_epoch,
                              nb_epoch=number_of_epoch,
                              validation_data=validation_generator,
                              nb_val_samples=val_samples_epoch,
                              callbacks=[
                                  ModelCheckpoint('./weights/test.hdf5',
                                                  monitor='val_acc',
                                                  save_best_only=True,
                                                  save_weights_only=True),
                                  TensorBoard(log_dir='./tf_logs/{}'.format(results_name)),
                                  EarlyStopping(monitor='val_acc', patience=30),
                              ])
print('Total training time: {:.2f} s'.format(time.time() - start_time))

print('\n--------------------------------')
print('FIRST EVALUATING PERFORMANCE ON VALIDATION SET')
print('--------------------------------\n')
result = model.evaluate_generator(validation_generator, val_samples=test_samples)
print('Loss: {:.2f} \t Accuracy: {:.2f} %'.format(result[0], result[1] * 100))

model.load_weights('./weights/test.hdf5')
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print('\n--------------------------------')
print('SECOND EVALUATING PERFORMANCE ON VALIDATION SET')
print('--------------------------------\n')
result = model.evaluate_generator(validation_generator, val_samples=test_samples)
print('Loss: {:.2f} \t Accuracy: {:.2f} %'.format(result[0], result[1] * 100))

print('\n--------------------------------')
print('STORING LOSS AND ACCURACY PLOTS')
print('--------------------------------\n')

# Store history
io.save_object(history.history, results_name, ignore=True)

# Plot
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim((0, 1))
plt.legend(['train', 'validation'], loc='lower right')
plt.savefig('./results/{}_accuracy.jpg'.format(results_name))
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Categorical cross-entropy (loss)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig('./results/{}_loss.jpg'.format(results_name))
plt.close()
