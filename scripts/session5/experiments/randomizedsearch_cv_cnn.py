from __future__ import print_function

import time

from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam
from keras.applications import VGG16
from keras.layers import MaxPooling2D, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model

from sklearn.model_selection import ParameterSampler
from scipy.stats.distributions import uniform

from mlcv.cnn import preprocess_input

""" CONSTANTS """
train_data_dir = '/data/MIT/train/'
val_data_dir = '/data/MIT/validation'
test_data_dir = '/data/MIT/test/'
img_width = 224
img_height = 224
batch_size = 32
samples_epoch = 1200
val_samples_epoch = 400
test_samples = 800
number_of_epoch_fc = 20
number_of_epoch_full = 10
n_iter = 100

""" HYPERPARAMETERS """
param_grid = {
    'batch_size': range(10, 100, 10),
    'optimizer': [SGD, RMSprop, Adagrad, Adadelta, Adam],
    'learning_rate': uniform(),
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
x = Dense(4096, activation='relu', name='fc')(x)
x = Dense(8, activation='softmax', name='predictions')(x)

# Create new model and save it
model = Model(input=base_model.input, output=x)

# Create data generator
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


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
    else:
        optimizer_instance = optimizer(lr=learning_rate)

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
                                     nb_val_samples=val_samples_epoch)
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
                                       nb_val_samples=val_samples_epoch)
    print('Total training time: {:.2f} s'.format(time.time() - start_time))

    print('\n--------------------------------')
    print('EVALUATING PERFORMANCE ON VALIDATION SET')
    print('--------------------------------\n')
    result = model.evaluate_generator(validation_generator, val_samples=val_samples_epoch)
    print('Loss: {:.2f} \t Accuracy: {:.2f} %'.format(result[0], result[1] * 100))

