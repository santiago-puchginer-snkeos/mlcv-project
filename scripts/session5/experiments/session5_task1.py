from __future__ import print_function, division

import matplotlib.pyplot as plt
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, MaxPooling2D, Convolution2D
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot
from keras.optimizers import Adadelta, SGD
from keras.regularizers import l2, activity_l2
import time

""" CONSTANTS """
train_data_dir = './dataset/400_dataset'
val_data_dir = './dataset/MIT_split/validation'
test_data_dir = './dataset/MIT_split/test'
img_width = 224
img_height = 224
batch_size = 20
number_of_epoch = 50


def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        # 'RGB'->'BGR'
        x = x[::-1, :, :]
        # Zero-center by mean pixel
        x[0, :, :] -= 103.939
        x[1, :, :] -= 116.779
        x[2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    return x


# create the base pre-trained model
base_model = VGG16(weights='imagenet')
plot(base_model, to_file='./results/modelVGG16a.png', show_shapes=True, show_layer_names=True)
# Get output from last convolutional layer in block 4

x = base_model.get_layer('block4_conv3').output
#x = MaxPooling2D(pool_size=(4, 4))(x)
#x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
#x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
#x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten(name='flat')(x)
x = Dense(4096, activation='relu', name='fc')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(8, activation='softmax', name='predictions')(x)

model = Model(input=base_model.input, output=x)
plot(model, to_file='./results/modelVGG16_task1.png', show_shapes=True, show_layer_names=True)
for layer in base_model.layers:
    layer.trainable = False


optimizer=SGD(lr=1e-5, momentum=0.9, decay=0.0, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
for layer in model.layers:
    print(layer.name, layer.trainable)

# preprocessing_function=preprocess_input,
datagen = ImageDataGenerator(featurewise_center=True,
                             samplewise_center=True,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=True,
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
                                             target_size=(img_width, img_height),
                                             batch_size=batch_size,
                                             class_mode='categorical')

validation_generator = datagen.flow_from_directory(val_data_dir,
                                                   target_size=(img_width, img_height),
                                                   batch_size=batch_size,
                                                   class_mode='categorical')

start_time = time.time()
history = model.fit_generator(train_generator,
                              samples_per_epoch=2000,
                              nb_epoch=number_of_epoch,
                              validation_data=validation_generator,
                              nb_val_samples=807)
print('Total training time: {:.2f} s'.format(time.time() - start_time))

print ('\nSaving weights into file...')
model.save_weights('./results/weights_DA_pool_flaten_2fc_predict')
print('\nEvaluating performance on test set...')
result = model.evaluate_generator(test_generator, val_samples=807)
print('Loss: {:.2f} \t Accuracy: {:.2f} %'.format(result[0], result[1]*100))

# list all data in history

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('./results/accuracy.jpg')
plt.close()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('./results/loss.jpg')
