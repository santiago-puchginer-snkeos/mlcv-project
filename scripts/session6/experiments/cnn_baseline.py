from __future__ import print_function, division

import time
import os
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers import Dense, MaxPooling2D, Flatten, Input, Convolution2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

import mlcv.input_output as io
from mlcv.cnn import preprocess_input

""" CONSTANTS """
train_data_dir = './dataset/MIT_split/train'
val_data_dir = './dataset/MIT_split/validation'
test_data_dir = './dataset/MIT_split/test'
img_width = 128
img_height = 128
samples_epoch = 2000
val_samples_epoch = 400
test_samples = 800
number_of_epoch = 150
results_name = os.path.basename(__file__).replace('.py', '')

# Hyperparameters
regularization = 0.01
batch_size = 40
lr = 1e-4
optimizer = Adam(lr=lr)

# Get output from last convolutional layer in block 4
x = Input(shape=(img_width, img_height, 3))
z = Convolution2D(32, 3, 3, init='he_normal', activation='relu', border_mode='same', W_regularizer=l2(regularization),
                  name='conv1')(x)
z = MaxPooling2D(pool_size=(2, 2), name='maxpooling1')(z)
z = Convolution2D(64, 3, 3, init='he_normal', activation='relu', border_mode='same', W_regularizer=l2(regularization),
                  name='conv2')(z)
z = MaxPooling2D(pool_size=(2, 2), name='maxpooling2')(z)
z = Convolution2D(64, 3, 3, init='he_normal', activation='relu', border_mode='same', W_regularizer=l2(regularization),
                  name='conv3')(z)
z = MaxPooling2D(pool_size=(2, 2), name='maxpooling3')(z)
z = Flatten()(z)
z = Dense(2048, activation='relu', W_regularizer=l2(regularization), name='fc')(z)
z = Dense(2048, activation='relu', W_regularizer=l2(regularization), name='fc2')(z)
y = Dense(8, activation='softmax', name='predictions')(z)

# Create new model and save it
model = Model(input=x, output=y)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print('\n{:^80}\n'.format('MODEL SUMMARY'))
model.summary()

# Data generators
datagen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True,
                             preprocessing_function=preprocess_input)

# Load train dataset and fit the ImageDataGenerator
print('Fitting ImageDataGenerator to the train dataset...')
train_images, train_labels = io.load_dataset_from_directory(train_data_dir)
datagen.fit(train_images)

# Create the generators
train_generator = datagen.flow_from_directory(train_data_dir,
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
                                                  save_best_only=True,
                                                  save_weights_only=True),
                                  TensorBoard(log_dir='./tf_logs/{}'.format(results_name)),
                                  EarlyStopping(monitor='val_loss', patience=5)
                              ])
print('Total training time: {:.2f} s'.format(time.time() - start_time))

print('\n--------------------------------')
print('EVALUATING PERFORMANCE ON VALIDATION SET')
print('--------------------------------\n')
result = model.evaluate_generator(validation_generator, val_samples=test_samples)
print('Loss: {:.2f} \t Accuracy: {:.2f} %'.format(result[0], result[1] * 100))

print('\n--------------------------------')
print('STORING LOSS AND ACCURACY PLOTS')
print('--------------------------------\n')

# Plot
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy (only FC layers training)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim((0, 1))
plt.legend(['train', 'validation'], loc='lower right')
plt.savefig('./results/{}_accuracy.jpg'.format(results_name))
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss (only FC layers training)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig('./results/{}_loss.jpg'.format(results_name))
plt.close()
