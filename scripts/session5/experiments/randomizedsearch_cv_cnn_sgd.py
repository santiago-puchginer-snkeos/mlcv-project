from __future__ import print_function

import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from keras.applications import VGG16
from keras.layers import Dense, Flatten, MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from scipy.stats.distributions import uniform
from sklearn.model_selection import ParameterSampler

from mlcv.cnn import preprocess_input

""" CONSTANTS """
train_data_dir = './dataset/400_dataset/'
val_data_dir = './dataset/MIT_split/validation'
test_data_dir = './dataset/MIT_split/test'
img_width = 224
img_height = 224
samples_epoch = 2000
val_samples_epoch = 400
test_samples = 800
number_of_epoch_fc = 10
number_of_epoch_full = 10
n_iter = 5

""" HYPERPARAMETERS """
param_grid = {
    'batch_size': range(5, 50, 5),
    'learning_rate': np.logspace(-7, -4, 10 ** 6),
    'momentum': uniform(),
    'nesterov': [True, False],
    'regularizer': np.logspace(-4, 0, 10 ** 5),
}

""" SETUP BASE MODEL """
# Get the base pre-trained model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Empty results dictionary
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
                             rescale=None,
                             preprocessing_function=preprocess_input)

""" RANDOMIZED SEARCH """
sampled_params = list(ParameterSampler(param_grid, n_iter=n_iter))
for ind, params in enumerate(sampled_params):

    print('\nITERATION {} of {}'.format(ind + 1, n_iter))

    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    regularizer = params['regularizer']
    momentum = params['momentum']
    nesterov = params['nesterov']

    print('batch size: {}'.format(batch_size))
    print('learning rate: {}'.format(learning_rate))
    print('regularization term: {}'.format(regularizer))
    print('momentum: {}'.format(momentum))
    print('nesterov momentum: {}'.format('Yes' if nesterov else 'No'))
    optimizer = SGD(lr=learning_rate, momentum=momentum, nesterov=nesterov)
    name = 'batchsize_{}_opt_{}_lr_{:.5G}_mom_{:.4G}_nesterov_{}'.format(
        batch_size,
        'sgd',
        learning_rate,
        momentum,
        str(nesterov).lower()
    )

    # Create generators
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
    print('FULLY CONNECTED LAYERS TRAINING')
    print('--------------------------------\n')
    start_time = time.time()

    # Create new model from base model
    x = base_model.get_layer('block4_conv3').output
    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Flatten(name='flat')(x)
    x = Dense(2048, activation='relu', name='fc', W_regularizer=l2(regularizer))(x)
    x = Dense(2048, activation='relu', name='fc2', W_regularizer=l2(regularizer))(x)
    x = Dense(8, activation='softmax', name='predictions')(x)
    model = Model(input=base_model.input, output=x)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history_fc = model.fit_generator(train_generator,
                                     samples_per_epoch=samples_epoch,
                                     nb_epoch=number_of_epoch_fc,
                                     validation_data=validation_generator,
                                     nb_val_samples=val_samples_epoch
                                     )
    model.save_weights('./weights/cnn_optimization_fc_{}.hdf5'.format(name))
    print('Total training time: {:.2f} s'.format(time.time() - start_time))

    print('\n--------------------------------')
    print('FULL NETWORK TRAINING')
    print('--------------------------------\n')
    start_time = time.time()
    # Unfreeze original model, recompile it and retrain it
    for layer in model.layers:
        layer.trainable = True
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history_full = model.fit_generator(train_generator,
                                       samples_per_epoch=samples_epoch,
                                       nb_epoch=number_of_epoch_full,
                                       validation_data=validation_generator,
                                       nb_val_samples=val_samples_epoch
                                       )

    model.save_weights('./weights/cnn_optimization_full_{}.hdf5'.format(name))
    print('Total training time: {:.2f} s'.format(time.time() - start_time))

    print('\n--------------------------------')
    print('EVALUATING PERFORMANCE ON VALIDATION SET')
    print('--------------------------------\n')
    result = model.evaluate_generator(validation_generator, val_samples=val_samples_epoch)
    print('Loss: {:.2f} \t Accuracy: {:.2f} %'.format(result[0], result[1] * 100))
    results_dict.update({
        name: {
            'accuracy': result[1] * 100,
            'loss': result[0]
        }
    })
    with open('./results/cnn_optimization_sgd.pickle', 'wb') as f:
        pickle.dump(results_dict, f)

    print('\n--------------------------------')
    print('STORING LOSS AND ACCURACY PLOTS')
    print('--------------------------------\n')
    plt.plot(history_fc.history['acc'])
    plt.plot(history_fc.history['val_acc'])
    plt.title('Model accuracy (only FC layers training)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./results/cnn_optimization_accuracy_fc' + name + '.jpg')
    plt.close()

    plt.plot(history_full.history['acc'])
    plt.plot(history_full.history['val_acc'])
    plt.title('Model accuracy (whole network training)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./results/cnn_optimization_accuracy_full' + name + '.jpg')
    plt.close()

    plt.plot(history_fc.history['loss'])
    plt.plot(history_fc.history['val_loss'])
    plt.title('Model loss (only FC layers training)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./results/cnn_optimization_loss_fc' + name + '.jpg')
    plt.close()

    plt.plot(history_full.history['loss'])
    plt.plot(history_full.history['val_loss'])
    plt.title('Model loss (whole network training)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./results/cnn_optimization_loss_full' + name + '.jpg')
    plt.close()
