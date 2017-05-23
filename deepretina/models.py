"""
Construct Keras models
"""

from __future__ import absolute_import, division, print_function
from keras.models import Sequential, Model
from keras.layers import Input, Reshape, Permute, TimeDistributed
from keras.layers.core import Dropout, Dense, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise, GaussianDropout
from keras.regularizers import l1_l2, l2
from keras import initializers
from .utils import notify
from .activations import ParametricSoftplus, ReQU

__all__ = ['sequential', 'ln', 'convnet', 'fixedlstm', 'generalizedconvnet', 'nips_conv', 'conv_rgcs']


def sequential(layers, optimizer, loss='poisson'):
    """Compiles a Keras model with the given layers

    Parameters
    ----------
    layers : list
        A list of Keras layers, in order

    optimizer : string or optimizer
        Either the name of a Keras optimizer, or a Keras optimizer object

    loss : string, optional
        The name of a Keras loss function (Default: 'poisson_loss'), or a
        Keras objective object

    Returns
    -------
    model : keras.models.Sequential
        A compiled Keras model object
    """
    model = Sequential(layers)
    with notify('Compiling'):
        model.compile(loss=loss, optimizer=optimizer)
    return model


def functional(inputs, outputs, optimizer, loss='poisson'):
    """Compiles a keras functional model

    Parameters
    ----------
    inputs: keras tensor
    outputs: keras tensor
    optimizer: Keras optimizer name or object
    loss: string, optional (default: poisson)
    """
    model = Model(inputs=inputs, outputs=outputs)
    with notify('Compiling'):
        model.compile(loss=loss, optimizer=optimizer)
    return model


def ln(input_shape, nout, weight_init='glorot_normal', l2_reg=0.0):
    """A linear-nonlinear model

    Parameters
    ----------
    input_shape : tuple
        The shape of the stimulus (e.g. (40,50,50))

    nout : int
        Number of output cells

    weight_init : string, optional
        Keras weight initialization (default: 'glorot_normal')

    l2_reg : float, optional
        l2 regularization on the weights (default: 0.0)
    """
    layers = list()
    layers.append(Flatten(input_shape=input_shape))
    layers.append(Dense(nout, init=weight_init, kernel_regularizer=l2(l2_reg)))
    layers.append(ParametricSoftplus())
    return layers


def nips_conv(num_cells):
    """Hard-coded model for NIPS"""
    layers = list()
    input_shape = (40, 50, 50)

    # injected noise strength
    sigma = 0.1

    # convolutional layer sizes
    convlayers = [(16, 15), (8, 9)]

    # l2_weight_regularization for every layer
    l2_weight = 1e-3

    # weight and activity regularization
    W_reg = [(0., l2_weight), (0., l2_weight)]
    act_reg = [(0., 0.), (0., 0.)]

    # loop over convolutional layers
    for (n, size), w_args, act_args in zip(convlayers, W_reg, act_reg):
        args = (n, size, size)
        kwargs = {
            'border_mode': 'valid',
            'subsample': (1, 1),
            'init': 'normal',
            'kernel_regularizer': l1_l2(*w_args),
            'activity_regularizer': l1_l2(*act_args),
        }
        if len(layers) == 0:
            kwargs['input_shape'] = input_shape

        # add convolutional layer
        layers.append(Conv2D(*args, **kwargs))

        # add gaussian noise
        layers.append(GaussianNoise(sigma))

        # add ReLu
        layers.append(Activation('relu'))

    # flatten
    layers.append(Flatten())

    # Add a final dense (affine) layer
    layers.append(Dense(num_cells, init='normal',
                        kernel_regularizer=l1_l2(0., l2_weight),
                        activity_regularizer=l1_l2(1e-3, 0.)))

    # Finish it off with a parameterized softplus
    layers.append(Activation('softplus'))

    return layers


def bn_layer(x, nchan, size, l2_reg, sigma=0.05, **kwargs):
    n = int(x.shape[-1]) - size + 1
    y = Conv2D(nchan, size, data_format="channels_first", kernel_regularizer=l2(l2_reg), **kwargs)(x)
<<<<<<< Updated upstream
    y = Reshape((nchan, n, n))(BatchNormalization(axis=-1)(Reshape((nchan, n ** 2))(y)))
=======
    y = Reshape((nchan, n, n))(BatchNormalization(axis=-1)(Reshape((nchan * n ** 2))(y)))
>>>>>>> Stashed changes
    return Activation('relu')(GaussianNoise(sigma)(y))


def td_conv(x, nchan, size, l2_reg, sigma=0.05, data_format="channels_first", **kwargs):
    n = int(x.shape[-1]) - size + 1
    y = TimeDistributed(Conv2D(nchan, size, data_format=data_format, kernel_regularizer=l2(l2_reg), **kwargs))(x)
    # y = Reshape((nchan, n, n))(BatchNormalization(axis=-1)(Reshape((nchan * n ** 2))(y)))
    return Activation('relu')(GaussianNoise(sigma)(y))


def conv_lstm_model(input_shape, nout, l2_reg=0.05):
    x = Input(shape=input_shape)
    u = td_conv(x, 8, 15, l2_reg, data_format="channels_first")
    z = conv_lstm(u, 2, data_format="channels_first", return_sequences=True)
    u = td_conv(x, 8, 11, l2_reg, data_format="channels_last")
    z = conv_lstm(u, 2, data_format="channels_last", return_sequences=True)
    y = Activation('softplus')(TimeDistributed(Dense(nout))(TimeDistributed(Flatten())(z)))
    return x, y


def conv_lstm(x, size, data_format="channels_first", return_sequences=True):
    """Convolutional LSTM"""

    if data_format=="channels_first":
        # x.shape is (T, C, X, Y)
        _, nt, nc, nx, ny = x.shape.as_list()
        y = Permute((3, 4, 1, 2))(x)
    elif data_format=="channels_last":
        # x.shape is (T, X, Y, C)
        _, nt, nx, ny, nc = x.shape.as_list()
        y = Permute((2, 3, 1, 4))(x)

    y = Reshape(target_shape=(nx * ny, nt, nc))(y)
    y = TimeDistributed(LSTM(size, return_sequences=return_sequences))(y)
    if return_sequences:
        y = Reshape(target_shape=(nt, nx, ny, size))(y)
    else:
        y = Reshape(target_shape=(nx, ny, size))(y)
    return y


def bn_cnn(input_shape, nout, l2_reg=0.05):

    x = Input(shape=input_shape)

    y = bn_layer(x, 8, 15, l2_reg, input_shape=input_shape)
    y = bn_layer(y, 8, 11, l2_reg)

    y = Dense(nout, use_bias=False)(Flatten()(y))
    y = BatchNormalization(axis=-1)(y)
    y = Activation('softplus')(y)
    # y = ReQU()(y)

    return x, y


def bn_cnn_requ(input_shape, nout, l2_reg):

    x = Input(shape=input_shape)

    y1 = Conv2D(8, 15, strides=(1, 1), input_shape=input_shape, data_format="channels_first", kernel_regularizer=l2(l2_reg))(x)
    y1 = BatchNormalization()(y1)
    y1 = GaussianNoise(0.05)(y1)
    y1 = Activation('relu')(y1)

    y2 = Conv2D(8, 11, strides=(1, 1), data_format="channels_first", kernel_regularizer=l2(l2_reg))(y1)
    y2 = BatchNormalization()(y2)
    y2 = GaussianNoise(0.05)(y2)
    y2 = Activation('relu')(y2)

    # y = Concatenate()([Flatten()(l2), Flatten()(l1)])
    y = Dense(nout)(Flatten()(y2))
    y = BatchNormalization()(y)
    y = ReQU()(y)

    return x, y


def cnn_bn_requ(input_shape, nout):

    x = Input(shape=input_shape)

    l1 = BatchNormalization(input_shape=input_shape)(x)
    l1 = Conv2D(8, 15, strides=(1, 1), data_format="channels_first")(l1)
    l1 = GaussianNoise(0.05)(l1)
    l1 = Activation('relu')(l1)

    l2 = BatchNormalization()(l1)
    l2 = Conv2D(8, 7, strides=(1, 1), data_format="channels_first")(l2)
    l2 = GaussianNoise(0.05)(l2)
    l2 = Activation('relu')(l2)

    # y = Concatenate()([Flatten()(l2), Flatten()(l1)])
    y = BatchNormalization()(l2)
    y = Dense(nout)(Flatten()(y))
    y = ReQU()(y)

    return x, y


def convnet(input_shape, nout,
            num_filters=(8, 16), filter_size=(13, 13),
            weight_init='normal',
            l2_reg_weights=(0.0, 0.0, 0.0),
            l1_reg_weights=(0.0, 0.0, 0.0),
            l2_reg_activity=(0.0, 0.0, 0.0),
            l1_reg_activity=(0.0, 0.0, 0.0),
            dropout=(0.0, 0.0)):
    """Convolutional neural network

    Parameters
    ----------
    input_shape : tuple
        The shape of the stimulus (e.g. (40,50,50))

    nout : int
        Number of output cells

    num_filters : tuple, optional
        Number of filters in each layer. Default: (8, 16)

    filter_size : tuple, optional
        Convolutional filter size. Default: (13, 13)

    weight_init : string, optional
        Keras weight initialization (default: 'normal')

    l2_weights: tuple of floats, optional
        l2 regularization on the weights for each layer (default: 0.0)

    l2_activity: tuple of floats, optional
        l2 regularization on the activations for each layer (default: 0.0)

    dropout : tuple of floats, optional
        Fraction of units to 'dropout' for regularization (default: 0.0)
    """
    layers = list()

    def _regularize(layer_idx):
        """Small helper function to define per layer regularization"""
        return {
            'kernel_regularizer': l1_l2(l1_reg_weights[layer_idx], l2_reg_weights[layer_idx]),
            'activity_regularizer': l1_l2(l1_reg_activity[layer_idx], l2_reg_activity[layer_idx]),
        }

    # first convolutional layer
    layers.append(Conv2D(num_filters[0], filter_size[0], filter_size[1],
                                input_shape=input_shape, init=weight_init,
                                border_mode='valid', subsample=(1, 1),
                                **_regularize(0)))

    # Add relu activation
    layers.append(Activation('relu'))

    # max pooling layer
    layers.append(MaxPooling2D(pool_size=(2, 2)))

    # flatten
    layers.append(Flatten())

    # Dropout (first stage)
    layers.append(Dropout(dropout[0]))

    # Add dense (affine) layer
    layers.append(Dense(num_filters[1], init=weight_init, **_regularize(1)))

    # Add relu activation
    layers.append(Activation('relu'))

    # Dropout (second stage)
    layers.append(Dropout(dropout[1]))

    # Add a final dense (affine) layer
    layers.append(Dense(nout, init=weight_init, **_regularize(2)))

    # Finish it off with a parameterized softplus
    layers.append(Activation('softplus'))

    return layers
