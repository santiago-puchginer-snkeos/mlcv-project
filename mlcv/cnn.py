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


""" MODEL 1: DEEP CONV """


def deepconv(img_width, img_height, regularization=0.1, batchnorm_after_activation=False, awgn_sigma=0,
             dropout=None):
    x = Input(shape=(img_width, img_height, 3), name='input')

    if awgn_sigma > 0:
        z = GaussianNoise(sigma=awgn_sigma)(x)
    else:
        z = x

    z = Convolution2D(32, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(regularization), name='conv1')(z)
    if not batchnorm_after_activation:
        z = BatchNormalization(name='batchnorm1')(z)
    z = Activation('relu')(z)
    if batchnorm_after_activation:
        z = BatchNormalization(name='batchnorm1')(z)

    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling1')(z)

    z = Convolution2D(64, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(regularization), name='conv2')(z)
    if not batchnorm_after_activation:
        z = BatchNormalization(name='batchnorm2')(z)
    z = Activation('relu')(z)
    if batchnorm_after_activation:
        z = BatchNormalization(name='batchnorm2')(z)

    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling2')(z)

    z = Convolution2D(64, 3, 3, init='he_normal', border_mode='same', W_regularizer=l2(regularization), name='conv3')(z)
    if not batchnorm_after_activation:
        z = BatchNormalization(name='batchnorm3')(z)
    z = Activation('relu')(z)
    if batchnorm_after_activation:
        z = BatchNormalization(name='batchnorm3')(z)

    z = MaxPooling2D(pool_size=(2, 2), name='maxpooling3')(z)

    z = Flatten()(z)
    z = Dense(2048, activation='relu', W_regularizer=l2(regularization), name='fc')(z)
    if dropout is not None:
        z = Dropout(dropout, name='dropout_fc')(z)
    z = Dense(2048, activation='relu', W_regularizer=l2(regularization), name='fc2')(z)
    if dropout is not None:
        z = Dropout(dropout, name='dropout_fc2')(z)
    y = Dense(8, activation='softmax', name='predictions')(z)

    model = Model(input=x, output=y)

    return model


""" MODEL 2: DENSEPARAMS """


def denseparams(img_width, img_height, regularization=0.1, init='uniform', awgn_sigma=0, dropout=None):
    x = Input(shape=(img_width, img_height, 3), name='input')

    if awgn_sigma > 0:
        z = GaussianNoise(sigma=awgn_sigma)(x)
    else:
        z = x

    z = Convolution2D(32, 5, 5, activation='relu', init=init, border_mode='same', W_regularizer=l2(regularization),
                      name='conv1')(z)
    z = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling1')(z)
    z = BatchNormalization()(z)

    z = Convolution2D(64, 5, 5, activation='relu', init=init, border_mode='same', W_regularizer=l2(regularization),
                      name='conv2')(z)
    z = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling2')(z)
    z = BatchNormalization()(z)

    z = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling3')(z)

    z = Flatten(name='flatten')(z)
    z = Dense(4096, activation='relu', W_regularizer=l2(regularization), name='fc')(z)
    if dropout is not None:
        z = Dropout(dropout, name='dropout_fc')(z)

    y = Dense(8, activation='softmax', name='predictions')(z)
    model = Model(input=x, output=y)

    return model


""" MODEL 3: COMPACTPARAMS"""


def compactparams(img_width, img_height, regularization=0.1, awgn_sigma=0, dropout=None):
    x = Input(shape=(img_width, img_height, 3), name='input')

    if awgn_sigma > 0:
        z = GaussianNoise(sigma=awgn_sigma)(x)
    else:
        z = x

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
    if dropout is not None:
        z = Dropout(dropout, name='dropout_fc')(z)
    z = Dense(512, activation='relu', W_regularizer=l2(regularization), name='fc2')(z)
    if dropout is not None:
        z = Dropout(dropout, name='dropout_fc2')(z)
    y = Dense(8, activation='softmax', name='predictions')(z)

    model = Model(input=x, output=y)

    return model


""" MODEL 4: DEEPFULLYCONV """


def deepfullyconv(img_width, img_height, regularization=0.1):
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


""" MODEL 5: COMPACTPARAMS_FULLYCONV"""


def compactparams_fullyconv(img_width, img_height, regularization=0.1, awgn_sigma=0, dropout=None):
    x = Input(shape=(img_width, img_height, 3), name='input')

    if awgn_sigma > 0:
        z = GaussianNoise(sigma=awgn_sigma)(x)
    else:
        z = x

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
    if dropout is not None:
        z = Dropout(dropout, name='dropout_fc')(z)
    z = Dense(512, activation='relu', W_regularizer=l2(regularization), name='fc2')(z)
    if dropout is not None:
        z = Dropout(dropout, name='dropout_fc2')(z)
    y = Dense(8, activation='softmax', name='predictions')(z)

    model = Model(input=x, output=y)

    return model
