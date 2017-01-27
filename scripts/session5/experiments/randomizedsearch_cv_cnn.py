from __future__ import print_function

import time
import matplotlib.pyplot as plt
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam
from keras.applications import VGG16
from keras.layers import MaxPooling2D, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import numpy as np
import mlcv.input_output as io
from sklearn.model_selection import ParameterSampler
from scipy.stats.distributions import uniform

from mlcv.cnn import preprocess_input

""" CONSTANTS """
train_data_dir = './dataset/400_dataset/'
val_data_dir = './dataset/MIT_split/test'
test_data_dir = './dataset/MIT_split/test'
img_width = 224
img_height = 224
batch_size = 32
samples_epoch = 2000
val_samples_epoch = 400
test_samples = 800
number_of_epoch_fc = 30
number_of_epoch_full = 30
n_iter = 100

""" HYPERPARAMETERS """
param_grid = {
    'batch_size': range(10, 60, 10),
    'optimizer': [SGD, RMSprop, Adagrad, Adadelta, Adam],
    'learning_rate': np.logspace(-7, -1, 10^6),
    'momentum': uniform(),      # Only for SGD
    'nesterov': [True, False]   # Only for SGD
}

""" SETUP MODEL """
# Get the base pre-trained model
base_model = VGG16(weights='imagenet')


# Get output from last convolutional layer in block 4
x = base_model.get_layer('block4_conv3').output
x = MaxPooling2D(pool_size=(4, 4))(x)
x = Flatten(name='flat')(x)
x = Dense(4096, activation='relu', name='fc',W_regularizer=l2(0.01))(x)
x = Dense(4096, activation='relu', name='fc2',W_regularizer=l2(0.01))(x)
x = Dense(8, activation='softmax', name='predictions')(x)

# Create new model and save it
model = Model(input=base_model.input, output=x)
results_dict = {}
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
                             rescale=None)


""" RANDOMIZED SEARCH CV """
sampled_params = list(ParameterSampler(param_grid, n_iter=n_iter))

for ind, params in enumerate(sampled_params):

    print('\nITERATION {} of {}'.format(ind + 1, n_iter))

    batch_size = params['batch_size']
    optimizer = params['optimizer']
    learning_rate = params['learning_rate']
    momentum = params['momentum']
    nesterov = params['nesterov']

    print('batch size: {}'.format(batch_size))
    print('optimizer: {}'.format(optimizer))
    print('learning rate: {}'.format(learning_rate))

    if optimizer == SGD:
        print('momentum: {}'.format(momentum))
        print('nesterov momentum: {}'.format('Yes' if nesterov else 'No'))
        optimizer_instance = optimizer(lr=learning_rate, momentum=momentum, nesterov=nesterov)
        name = '_batchsize_{}_opt_{}_lr_{}_mom_{}_nesterov_{}'.format(batch_size, optimizer, learning_rate, momentum, nesterov)
    else:
        optimizer_instance = optimizer(lr=learning_rate)
        name = '_batchsize_{}_opt_{}_lr_{}'.format(batch_size, optimizer, learning_rate)

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
    # Freeze layers from VGG model, compile it and train
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_instance, metrics=['accuracy'])
    history_fc = model.fit_generator(train_generator,
                                     samples_per_epoch=samples_epoch,
                                     nb_epoch=number_of_epoch_fc,
                                     validation_data=validation_generator,
                                     nb_val_samples=val_samples_epoch, callbacks=[
                                     ModelCheckpoint('./weights/cnn_optimization_fc_'+name+'.{epoch:02d}.hdf5', save_best_only=True,
                                                     save_weights_only=True)])
    print('Total training time: {:.2f} s'.format(time.time() - start_time))

    print('\n--------------------------------')
    print('FULL NETWORK TRAINING')
    print('--------------------------------\n')
    start_time = time.time()
    # Unfreeze original model, recompile it and retrain it
    for layer in base_model.layers:
        layer.trainable = True
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_instance, metrics=['accuracy'])
    history_full = model.fit_generator(train_generator,
                                       samples_per_epoch=samples_epoch,
                                       nb_epoch=number_of_epoch_full,
                                       validation_data=validation_generator,
                                       nb_val_samples=val_samples_epoch, callbacks=[
                                     ModelCheckpoint('./weights/cnn_optimization_full_'+name+'.{epoch:02d}.hdf5', save_best_only=True,
                                                     save_weights_only=True)])
    print('Total training time: {:.2f} s'.format(time.time() - start_time))
    #io.save_object(history_fc, 'history_fc_'+name)
    #io.save_object(history_full, 'history_full_' + name)
    print('\n--------------------------------')
    print('EVALUATING PERFORMANCE ON VALIDATION SET')
    print('--------------------------------\n')
    result = model.evaluate_generator(validation_generator, val_samples=val_samples_epoch)
    print('Loss: {:.2f} \t Accuracy: {:.2f} %'.format(result[0], result[1] * 100))
    results_dict.update({name: {'accuracy': result[1]*100,
                             'loss': result[0]}})
    #
    io.save_object(results_dict, 'cnn_optimization', ignore=True)
    # print('\n--------------------------------')
    # print('STORING LOSS AND ACCURACY PLOTS')
    # print('--------------------------------\n')
    # plt.plot(history_fc.history['acc'])
    # plt.plot(history_fc.history['val_acc'])
    # plt.title('Model accuracy (only FC layers training)')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.savefig('./results/cnn_optimization_accuracy_fc'+name+'.jpg')
    # plt.close()
    #
    # plt.plot(history_full.history['acc'])
    # plt.plot(history_full.history['val_acc'])
    # plt.title('Model accuracy (whole network training)')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.savefig('./results/cnn_optimization_accuracy_full'+name+'.jpg')
    # plt.close()
    #
    # plt.plot(history_fc.history['loss'])
    # plt.plot(history_fc.history['val_loss'])
    # plt.title('Model loss (only FC layers training)')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.savefig('./results/cnn_optimization_loss_fc'+name+'.jpg')
    # plt.close()
    #
    # plt.plot(history_full.history['loss'])
    # plt.plot(history_full.history['val_loss'])
    # plt.title('Model loss (whole network training)')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.savefig('./results/cnn_optimization_loss_full'+name+'.jpg')
    # plt.close()
    #