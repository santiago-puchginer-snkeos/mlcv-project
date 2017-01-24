from __future__ import print_function, division

import math
import time

import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, MaxPooling2D, Convolution2D
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard

from mlcv.cnn import preprocess_input

""" CONSTANTS """
train_data_dir = './dataset/400_dataset'
val_data_dir = './dataset/MIT_split/test'
test_data_dir = './dataset/MIT_split/test'
img_width = 224
img_height = 224
batch_size = 32
samples_epoch = 1200
val_samples_epoch = 400
test_samples = 800
number_of_epoch_fc = 100
number_of_epoch_full = 100
lr = 1e-7


def learning_rate_scheduler(epoch):
    return lr / (math.ceil(epoch / 5) + 1)


# Get the base pre-trained model
base_model = VGG16(weights='imagenet')

# Get output from last convolutional layer in block 4
x = base_model.get_layer('block4_conv3').output
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', init='he_normal')(x)
x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', init='he_normal')(x)
x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', init='he_normal')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten(name='flat')(x)
x = Dense(4096, activation='relu', name='fc')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(8, activation='softmax', name='predictions')(x)

# Create new model and save it
model = Model(input=base_model.input, output=x)
plot(model, to_file='./results/finetune_cnn_maxpool2.png', show_shapes=True, show_layer_names=True)

datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             rotation_range=0.,
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
optimizer = SGD(lr=lr, momentum=0.9, decay=0.1, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optimizer,
              metrics=['accuracy', 'precision', 'recall', 'fbeta_score'])
history_fc = model.fit_generator(train_generator,
                                 samples_per_epoch=samples_epoch,
                                 nb_epoch=number_of_epoch_fc,
                                 validation_data=validation_generator,
                                 nb_val_samples=val_samples_epoch,
                                 callbacks=[
                                     ModelCheckpoint('./weights/cnn_maxpool2_fc.{epoch:02d}.hdf5', save_best_only=True,
                                                     save_weights_only=True),
                                     TensorBoard(log_dir='./tf_logs/cnn_maxpool2_fc'),
                                     LearningRateScheduler(learning_rate_scheduler),
                                     EarlyStopping(monitor='loss', patience=5)
                                 ]
                                 )
print('Total training time: {:.2f} s'.format(time.time() - start_time))

# Unfreeze original model and retrain it
for layer in base_model.layers:
    layer.trainable = True

print('\n--------------------------------')
print('FULL NETWORK TRAINING')
print('--------------------------------\n')
start_time = time.time()
for layer in base_model.layers:
    layer.trainable = True
optimizer = SGD(lr=lr, momentum=0.9, decay=0.1, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optimizer,
              metrics=['accuracy', 'precision', 'recall', 'fbeta_score'])
history_full = model.fit_generator(train_generator,
                                   samples_per_epoch=samples_epoch,
                                   nb_epoch=number_of_epoch_full,
                                   validation_data=validation_generator,
                                   nb_val_samples=val_samples_epoch,
                                   callbacks=[
                                       ModelCheckpoint('./weights/cnn_maxpool2_full.{epoch:02d}.hdf5',
                                                       save_best_only=True,
                                                       save_weights_only=True),
                                       TensorBoard(log_dir='./tf_logs/cnn_maxpool2_full'),
                                       LearningRateScheduler(learning_rate_scheduler),
                                       EarlyStopping(monitor='loss', patience=5)
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
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('./results/finetune_maxpool2_accuracy_fc.jpg')
plt.close()

plt.plot(history_full.history['acc'])
plt.plot(history_full.history['val_acc'])
plt.title('Model accuracy (whole network training)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('./results/finetune_maxpool2_accuracy_full.jpg')
plt.close()

plt.plot(history_fc.history['loss'])
plt.plot(history_fc.history['val_loss'])
plt.title('Model loss (only FC layers training)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('./results/finetune_maxpool2_loss_fc.jpg')

plt.plot(history_full.history['loss'])
plt.plot(history_full.history['val_loss'])
plt.title('Model loss (whole network training)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('./results/finetune_maxpool2_loss_full.jpg')

plt.plot(history_fc.history['fbeta_score'])
plt.plot(history_fc.history['val_fbeta_score'])
plt.title('F1 score (only FC layers training)')
plt.ylabel('F1')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('./results/finetune_maxpool2_fscore_fc.jpg')
plt.close()

plt.plot(history_full.history['fbeta_score'])
plt.plot(history_full.history['val_fbeta_score'])
plt.title('F1 score (whole network training)')
plt.ylabel('F1')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('./results/finetune_maxpool2_fscore_full.jpg')
plt.close()
