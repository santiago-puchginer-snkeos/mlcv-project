from __future__ import print_function, division

import os
import time

import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
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
number_of_epoch = 100
batch_size = 150

# Optimizer
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=10 ** (-4))

# Hyperparameters
regularization = 0.1
bn_aa = False
awgn_sigma = 0
dropout = 0

results_name = '{}_reg-{}_bnaa-{}_awgn-{}_dropout-{}'.format(
    os.path.basename(__file__).replace('.py', ''),
    regularization,
    bn_aa,
    awgn_sigma,
    dropout
)

print()
print(results_name.upper())
print()

# Create new model and save it
model = cnn.deepconv(img_width, img_height,
                     regularization=regularization,
                     batchnorm_after_activation=bn_aa,
                     awgn_sigma=awgn_sigma,
                     dropout=dropout
                     )
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
                                  ModelCheckpoint('./weights/{}.hdf5'.format(results_name),
                                                  monitor='val_acc',
                                                  mode='max',
                                                  save_best_only=True,
                                                  save_weights_only=False,
                                                  period=1,
                                                  verbose=1),
                                  TensorBoard(log_dir='./tf_logs/{}'.format(results_name)),
                                  EarlyStopping(monitor='val_loss', patience=10),
                              ])
print('Total training time: {:.2f} s'.format(time.time() - start_time))

print('\n--------------------------------')
print('EVALUATING PERFORMANCE ON VALIDATION SET (LAST EPOCH WEIGHTS)')
print('--------------------------------\n')
result = model.evaluate_generator(validation_generator, val_samples=test_samples)
print('Loss: {:.2f} \t Accuracy: {:.2f} %'.format(result[0], result[1] * 100))

print('\n--------------------------------')
print('EVALUATING PERFORMANCE ON VALIDATION SET (STORED WEIGHTS)')
print('--------------------------------\n')
new_model = load_model('./weights/{}.hdf5'.format(results_name))
result = new_model.evaluate_generator(validation_generator, val_samples=test_samples)
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
