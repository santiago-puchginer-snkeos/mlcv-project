import keras.backend as K

from keras.layers import Dense, MaxPooling2D, Flatten, Input, Convolution2D, BatchNormalization, Activation, \
    GaussianNoise, Dropout
from keras.models import Model
from keras.regularizers import l2


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


def baseline_cnn(img_width, img_height, regularization=0.1):
    x = Input(shape=(img_width, img_height, 3), name='input')
    z = Convolution2D(32, 3, 3, init='he_normal', activation='relu', border_mode='same',
                      W_regularizer=l2(regularization),
                      name='conv1')(x)
    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling1')(z)
    z = Convolution2D(64, 3, 3, init='he_normal', activation='relu', border_mode='same',
                      W_regularizer=l2(regularization),
                      name='conv2')(z)
    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling2')(z),
    z = Convolution2D(64, 3, 3, init='he_normal', activation='relu', border_mode='same',
                      W_regularizer=l2(regularization),
                      name='conv3')(z)
    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling3')(z)
    z = Flatten()(z)
    z = Dense(2048, activation='relu', W_regularizer=l2(regularization), name='fc')(z)
    z = Dense(2048, activation='relu', W_regularizer=l2(regularization), name='fc2')(z)
    y = Dense(8, activation='softmax', name='predictions')(z)

    model = Model(input=x, output=y)

    return model


def baseline_cnn_alt(img_width, img_height):
    x = Input(shape=(img_width, img_height, 3), name='input')

    z = Convolution2D(32, 5, 5, activation='relu', border_mode='same', name='conv1')(x)
    z = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling1')(z)
    z = BatchNormalization()(z)

    z = Convolution2D(64, 5, 5, activation='relu', border_mode='same', name='conv2')(z)
    z = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling2')(z)
    z = BatchNormalization()(z)

    z = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling3')(z)

    z = Flatten(name='flatten')(z)
    z = Dense(4096, activation='relu', name='fc')(z)
    y = Dense(8, activation='softmax', name='predictions')(z)

    model = Model(input=x, output=y)

    return model


def baseline_cnn_alt_mod(img_width, img_height, regularization=0.1):
    x = Input(shape=(img_width, img_height, 3), name='input')

    z = Convolution2D(32, 5, 5, init='he_normal', activation='relu', border_mode='same',
                      W_regularizer=l2(regularization), name='conv1')(x)
    z = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling1')(z)
    z = BatchNormalization()(z)

    z = Convolution2D(64, 5, 5, init='he_normal', activation='relu', border_mode='same',
                      W_regularizer=l2(regularization), name='conv2')(z)
    z = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling2')(z)
    z = BatchNormalization()(z)

    z = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling3')(z)

    z = Flatten(name='flatten')(z)
    z = Dense(4096, activation='relu', W_regularizer=l2(regularization), name='fc')(z)
    y = Dense(8, activation='softmax', name='predictions')(z)

    model = Model(input=x, output=y)

    return model


def baseline_cnn_alt_mod2(img_width, img_height):
    x = Input(shape=(img_width, img_height, 3), name='input')

    z = Convolution2D(32, 5, 5, init='glorot_normal', activation='relu', border_mode='same', name='conv1')(x)
    z = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling1')(z)
    z = BatchNormalization()(z)

    z = Convolution2D(64, 5, 5, init='glorot_normal', activation='relu', border_mode='same', name='conv2')(z)
    z = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling2')(z)
    z = BatchNormalization()(z)

    z = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling3')(z)

    z = Flatten(name='flatten')(z)
    z = Dense(4096, activation='relu', name='fc')(z)
    y = Dense(8, activation='softmax', name='predictions')(z)

    model = Model(input=x, output=y)

    return model


def baseline_cnn_alt_mod3(img_width, img_height, regularization=0.01):
    x = Input(shape=(img_width, img_height, 3), name='input')

    z = Convolution2D(32, 5, 5, activation='relu', border_mode='same',
                      W_regularizer=l2(regularization), name='conv1')(x)
    z = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling1')(z)
    z = BatchNormalization()(z)

    z = Convolution2D(64, 5, 5, init='he_normal', activation='relu', border_mode='same',
                      W_regularizer=l2(regularization), name='conv2')(z)
    z = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling2')(z)
    z = BatchNormalization()(z)

    z = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling3')(z)

    z = Flatten(name='flatten')(z)
    z = Dense(4096, activation='relu', W_regularizer=l2(regularization), name='fc')(z)
    y = Dense(8, activation='softmax', name='predictions')(z)

    model = Model(input=x, output=y)

    return model


def baseline_cnn_alt_awgn(img_width, img_height, regularization=0.01, sigma=0.05):
    x = Input(shape=(img_width, img_height, 3), name='input')
    z = GaussianNoise(sigma=sigma)(x)
    z = Convolution2D(32, 5, 5, activation='relu', border_mode='same',
                      W_regularizer=l2(regularization), name='conv1')(z)

    z = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling1')(z)
    z = BatchNormalization()(z)

    z = Convolution2D(64, 5, 5, init='he_normal', activation='relu', border_mode='same',
                      W_regularizer=l2(regularization), name='conv2')(z)
    z = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling2')(z)
    z = BatchNormalization()(z)

    z = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling3')(z)

    z = Flatten(name='flatten')(z)
    z = Dense(4096, activation='relu', W_regularizer=l2(regularization), name='fc')(z)
    y = Dense(8, activation='softmax', name='predictions')(z)

    model = Model(input=x, output=y)

    return model


def baseline_cnn_awgn_nobatchnorm(img_width, img_height, regularization=0.1, sigma=0.05):
    x = Input(shape=(img_width, img_height, 3), name='input')
    z = GaussianNoise(sigma=sigma)(x)
    z = Convolution2D(32, 3, 3, init='he_normal', activation='relu', border_mode='same',
                      W_regularizer=l2(regularization),
                      name='conv1')(z)
    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling1')(z)
    z = Convolution2D(64, 3, 3, init='he_normal', activation='relu', border_mode='same',
                      W_regularizer=l2(regularization),
                      name='conv2')(z)
    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling2')(z)
    z = Convolution2D(64, 3, 3, init='he_normal', activation='relu', border_mode='same',
                      W_regularizer=l2(regularization),
                      name='conv3')(z)
    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling3')(z)
    z = Flatten()(z)
    z = Dense(2048, activation='relu', W_regularizer=l2(regularization), name='fc')(z)
    z = Dense(2048, activation='relu', W_regularizer=l2(regularization), name='fc2')(z)
    y = Dense(8, activation='softmax', name='predictions')(z)

    model = Model(input=x, output=y)

    return model


def baseline_cnn_batchnorm(img_width, img_height, regularization=0.1, after_activation=True):
    x = Input(shape=(img_width, img_height, 3), name='input')

    z = Convolution2D(32, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(regularization), name='conv1')(x)
    if not after_activation:
        z = BatchNormalization(name='batchnorm1')(z)
    z = Activation('relu')(z)
    if after_activation:
        z = BatchNormalization(name='batchnorm1')(z)

    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling1')(z)

    z = Convolution2D(64, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(regularization), name='conv2')(z)
    if not after_activation:
        z = BatchNormalization(name='batchnorm2')(z)
    z = Activation('relu')(z)
    if after_activation:
        z = BatchNormalization(name='batchnorm2')(z)

    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling2')(z)

    z = Convolution2D(64, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(regularization), name='conv3')(z)
    if not after_activation:
        z = BatchNormalization(name='batchnorm3')(z)
    z = Activation('relu')(z)
    if after_activation:
        z = BatchNormalization(name='batchnorm3')(z)

    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling3')(z)

    z = Flatten()(z)
    z = Dense(2048, activation='relu', W_regularizer=l2(regularization), name='fc')(z)
    z = Dense(2048, activation='relu', W_regularizer=l2(regularization), name='fc2')(z)
    y = Dense(8, activation='softmax', name='predictions')(z)

    model = Model(input=x, output=y)

    return model


def baseline_cnn_awgn(img_width, img_height, regularization=0.1, sigma=0.05):
    x = Input(shape=(img_width, img_height, 3), name='input')
    z = GaussianNoise(sigma=sigma)(x)

    z = Convolution2D(32, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(regularization), name='conv1')(z)
    z = BatchNormalization(name='batchnorm1')(z)
    z = Activation('relu')(z)

    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling1')(z)

    z = Convolution2D(64, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(regularization), name='conv2')(z)
    z = BatchNormalization(name='batchnorm2')(z)
    z = Activation('relu')(z)

    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling2')(z)

    z = Convolution2D(64, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(regularization), name='conv3')(z)
    z = BatchNormalization(name='batchnorm3')(z)
    z = Activation('relu')(z)

    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling3')(z)

    z = Flatten()(z)
    z = Dense(2048, activation='relu', W_regularizer=l2(regularization), name='fc')(z)
    z = Dense(2048, activation='relu', W_regularizer=l2(regularization), name='fc2')(z)
    y = Dense(8, activation='softmax', name='predictions')(z)

    model = Model(input=x, output=y)

    return model


def baseline_cnn_alt_dropout_fc(img_width, img_height, regularization=0.01, dropout=0.5, gaussian_noise=False,
                                sigma=0.05):
    x = Input(shape=(img_width, img_height, 3), name='input')

    if gaussian_noise:
        z = GaussianNoise(sigma=sigma)(x)
    else:
        z = x
    z = Convolution2D(32, 5, 5, activation='relu', border_mode='same',
                      W_regularizer=l2(regularization), name='conv1')(z)
    z = MaxPooling2D(pool_size=(4, 4), strides=(2, 2), name='maxpooling1')(z)
    z = BatchNormalization()(z)

    z = Convolution2D(64, 5, 5, init='he_normal', activation='relu', border_mode='same',
                      W_regularizer=l2(regularization), name='conv2')(z)
    z = MaxPooling2D(pool_size=(6, 6), strides=(2, 2), name='maxpooling2')(z)
    z = BatchNormalization()(z)

    z = MaxPooling2D(pool_size=(6, 6), strides=(2, 2), name='maxpooling3')(z)

    z = Flatten(name='flatten')(z)
    z = Dense(4096, activation='relu', W_regularizer=l2(regularization), name='fc')(z)
    z = Dropout(dropout, name='dropout_fc')(z)
    y = Dense(8, activation='softmax', name='predictions')(z)

    model = Model(input=x, output=y)

    return model


def baseline_cnn_dropout_fc(img_width, img_height, regularization=0.1, dropout=0.5, gaussian_noise=False, sigma=0.05):
    x = Input(shape=(img_width, img_height, 3), name='input')

    if gaussian_noise:
        x = GaussianNoise(sigma=sigma)(x)

    z = Convolution2D(32, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(regularization), name='conv1')(x)
    z = BatchNormalization(name='batchnorm1')(z)
    z = Activation('relu')(z)

    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling1')(z)

    z = Convolution2D(64, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(regularization), name='conv2')(z)
    z = BatchNormalization(name='batchnorm2')(z)
    z = Activation('relu')(z)

    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling2')(z)

    z = Convolution2D(64, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(regularization), name='conv3')(z)
    z = BatchNormalization(name='batchnorm3')(z)
    z = Activation('relu')(z)

    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling3')(z)

    z = Flatten()(z)
    z = Dense(2048, activation='relu', W_regularizer=l2(regularization), name='fc')(z)
    z = Dropout(dropout, name='dropout_fc')(z)
    z = Dense(2048, activation='relu', W_regularizer=l2(regularization), name='fc2')(z)
    z = Dropout(dropout, name='dropout_fc2')(z)
    y = Dense(8, activation='softmax', name='predictions')(z)

    model = Model(input=x, output=y)

    return model


def baseline_cnn_alt_dropout_fc_mod(img_width, img_height, regularization=0.01, dropout=0.5, gaussian_noise=False, sigma=0.05):
    x = Input(shape=(img_width, img_height, 3), name='input')

    if gaussian_noise:
        z = GaussianNoise(sigma=sigma)(x)
    else:
        z=x
    z = Convolution2D(32, 5, 5, activation='relu', border_mode='same',
                      W_regularizer=l2(regularization), name='conv1')(z)
    z = MaxPooling2D(pool_size=(4, 4), strides=(2, 2), name='maxpooling1')(z)
    z = BatchNormalization()(z)

    z = Convolution2D(64, 5, 5, activation='relu', border_mode='same',
                      W_regularizer=l2(regularization), name='conv2')(z)
    z = MaxPooling2D(pool_size=(4, 4), strides=(2, 2), name='maxpooling2')(z)
    z = BatchNormalization()(z)

    z = MaxPooling2D(pool_size=(4, 4), strides=(2, 2), name='maxpooling3')(z)

    z = Flatten(name='flatten')(z)
    z = Dense(512, activation='relu', W_regularizer=l2(regularization), name='fc')(z)
    z = Dropout(dropout, name='dropout_fc')(z)
    z = Dense(512, activation='relu', W_regularizer=l2(regularization), name='fc2')(z)
    z = Dropout(dropout, name='dropout_fc2')(z)
    y = Dense(8, activation='softmax', name='predictions')(z)

    model = Model(input=x, output=y)

    return model



def baseline_cnn_dropout_conv(img_width, img_height, regularization=0.1, dropout=0.5):
    x = Input(shape=(img_width, img_height, 3), name='input')

    z = Convolution2D(32, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(regularization), name='conv1')(x)
    z = BatchNormalization(name='batchnorm1')(z)
    z = Activation('relu')(z)
    z = Dropout(dropout, name='dropout_conv1')(z)

    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling1')(z)

    z = Convolution2D(64, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(regularization), name='conv2')(z)
    z = BatchNormalization(name='batchnorm2')(z)
    z = Activation('relu')(z)
    z = Dropout(dropout, name='dropout_conv2')(z)

    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling2')(z)

    z = Convolution2D(64, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(regularization), name='conv3')(z)
    z = BatchNormalization(name='batchnorm3')(z)
    z = Activation('relu')(z)
    z = Dropout(dropout, name='dropout_conv3')(z)

    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling3')(z)

    z = Flatten()(z)
    z = Dense(2048, activation='relu', W_regularizer=l2(regularization), name='fc')(z)
    z = Dense(2048, activation='relu', W_regularizer=l2(regularization), name='fc2')(z)
    y = Dense(8, activation='softmax', name='predictions')(z)

    model = Model(input=x, output=y)

    return model


def baseline_cnn_dropout(img_width, img_height, regularization=0.1, sigma=0.05, dropout_conv=0.5, dropout_fc=0.5):
    x = Input(shape=(img_width, img_height, 3), name='input')
    z = GaussianNoise(sigma=sigma)(x)

    z = Convolution2D(32, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(regularization), name='conv1')(z)
    z = BatchNormalization(name='batchnorm1')(z)
    z = Activation('relu')(z)
    z = Dropout(dropout_conv, name='dropout_conv1')(z)

    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling1')(z)

    z = Convolution2D(64, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(regularization), name='conv2')(z)
    z = BatchNormalization(name='batchnorm2')(z)
    z = Activation('relu')(z)
    z = Dropout(dropout_conv, name='dropout_conv2')(z)

    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling2')(z)

    z = Convolution2D(64, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(regularization), name='conv3')(z)
    z = BatchNormalization(name='batchnorm3')(z)
    z = Activation('relu')(z)
    z = Dropout(dropout_conv, name='dropout_conv3')(z)

    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling3')(z)

    z = Flatten()(z)
    z = Dense(2048, activation='relu', W_regularizer=l2(regularization), name='fc')(z)
    z = Dropout(dropout_fc, name='dropout_fc')(z)
    z = Dense(2048, activation='relu', W_regularizer=l2(regularization), name='fc2')(z)
    z = Dropout(dropout_fc, name='dropout_fc2')(z)
    y = Dense(8, activation='softmax', name='predictions')(z)

    model = Model(input=x, output=y)

    return model


""" FULLY CONVOLUTIONAL MODELS """


def cnn_baseline_fully_conv(img_width, img_height, regularization=0.1):
    x = Input(shape=(img_width, img_height, 3), name='input')
    z = Convolution2D(32, 3, 3, init='he_normal', activation='relu', border_mode='same',
                      W_regularizer=l2(regularization),
                      name='conv1')(x)
    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling1')(z)
    z = Convolution2D(64, 3, 3, init='he_normal', activation='relu', border_mode='same',
                      W_regularizer=l2(regularization),
                      name='conv2')(z)
    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling2')(z),
    z = Convolution2D(64, 3, 3, init='he_normal', activation='relu', border_mode='same',
                      W_regularizer=l2(regularization),
                      name='conv3')(z)
    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling3')(z)
    z = Convolution2D(2048, 16, 16, init='he_normal', activation='relu',
                      W_regularizer=l2(regularization), name='fclike_conv1')(z)
    z = Convolution2D(2048, 1, 1, init='he_normal', activation='relu',
                      W_regularizer=l2(regularization), name='fclike_conv2')(z)
    z = Convolution2D(8, 1, 1, init='he_normal', W_regularizer=l2(regularization), name='fclike_conv3')(z)
    z = Flatten()(z)
    y = Activation('softmax')(z)

    model = Model(input=x, output=y)

    return model


def cnn_baseline_alt_fully_conv(img_width, img_height, init='he_normal', regularization=0.1):
    x = Input(shape=(img_width, img_height, 3), name='input')

    z = Convolution2D(32, 5, 5, init=init, activation='relu', border_mode='same',
                      W_regularizer=l2(regularization), name='conv1')(x)
    z = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling1')(z)
    z = BatchNormalization()(z)

    z = Convolution2D(64, 5, 5, init=init, activation='relu', border_mode='same',
                      W_regularizer=l2(regularization), name='conv2')(z)
    z = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling2')(z)
    z = BatchNormalization()(z)

    z = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling3')(z)

    z = Convolution2D(4096, 16, 16, init=init, activation='relu', W_regularizer=l2(regularization),
                      name='fclike_conv1')(z)
    z = Convolution2D(8, 1, 1, init=init, W_regularizer=l2(regularization), name='fclike_conv2')(z)
    z = Flatten()(z)
    y = Activation('softmax')(z)

    model = Model(input=x, output=y)

    return model
