"""
Construct Keras models
"""

from __future__ import absolute_import, division, print_function
from keras.models import Sequential, Graph
from keras.layers.core import Dropout, Dense, Activation, Flatten, TimeDistributedDense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.advanced_activations import PReLU, ParametricSoftplus
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise, GaussianDropout
from keras.regularizers import l1l2, activity_l1l2, l2
from keras.layers.extra import TimeDistributedFlatten, TimeDistributedConvolution2D, TimeDistributedMaxPooling2D, SubunitSlice, ConvRNN, ImgReshape
from .utils import notify

__all__ = ['sequential', 'ln', 'convnet', 'fixedlstm', 'experimentallstm', 'generalizedconvnet']


def sequential(layers, optimizer, loss='poisson_loss'):
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
    model = Sequential()
    [model.add(layer) for layer in layers]
    with notify('Compiling'):
        model.compile(loss=loss, optimizer=optimizer)
    return model

def ln(input_shape, nout, weight_init='glorot_normal', l2_reg=0.0):
    """A linear-nonlinear stack of layers

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
    layers.append(Dense(nout, init=weight_init, W_regularizer=l2(l2_reg)))
    layers.append(ParametricSoftplus())
    return layers


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
            'W_regularizer': l1l2(l1_reg_weights[layer_idx], l2_reg_weights[layer_idx]),
            'activity_regularizer': activity_l1l2(l1_reg_activity[layer_idx], l2_reg_activity[layer_idx]),
        }

    # first convolutional layer
    layers.append(Convolution2D(num_filters[0], filter_size[0], filter_size[1],
                                input_shape=input_shape, init=weight_init,
                                border_mode='same', subsample=(1, 1),
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
    layers.append(ParametricSoftplus())

    return layers


def fixedlstm(input_shape, nout, num_hidden=1600, weight_init='he_normal', l2_reg=0.0):
    """LSTM network with fixed input (e.g. input from the CNN output)

    Parameters
    ----------
    input_shape : tuple
        The shape of the stimulus e.g. (num_timesteps, stimulus.ndim)

    nout : int
        Number of output cells

    num_timesteps : int
        Number of timesteps of history to include in the LSTM layer

    num_filters : int, optional
        Number of filters in input. (Default: 16)

    num_hidden : int, optional
        Number of hidden units in the LSTM layer (Default: 1600)

    weight_init : string, optional
        Weight initialization for the final Dense layer (default: 'normal')

    l2_reg : float, optional
        l2 regularization on the weights (default: 0.0)
    """
    layers = list()

    # Optional: Add relu activation separately
    # layers.append(Activation('relu', input_shape=input_shape))

    # add the LSTM layer
    layers.append(LSTM(num_hidden, return_sequences=False, input_shape=input_shape))

    # Add a final dense (affine) layer with softplus activation
    layers.append(Dense(nout,
                        init=weight_init,
                        W_regularizer=l2(l2_reg),
                        activation='softplus'))

    return layers

def experimentallstm(input_shape, nout, num_hidden=50,
            num_filters=(8, 16), filter_size=(13, 13),
            optimizer='adam', loss='poisson_loss', rnn_flag=3, weight_init='normal', reg=0.01):
    """Experimental Convolutional-LSTM neural network

    Parameters
    ----------
    input_shape : tuple
        The shape of the stimulus (e.g. (time, 40,50,50))

    nout : int
        Number of output cells

    num_filters : tuple, optional
        Number of filters in each layer. Default: (8, 16)

    filter_size : tuple, optional
        Convolutional filter size. Default: (13, 13)

    weight_init : string, optional
        Keras weight initialization (default: 'normal')

    reg: Just does l2 reg, can make this choice fancier and more modular later on
    rnn_flag: int either 1,2,or 3 where 1: SimpleRNN, 2: LSTM, 3: ConvRNN
    """

    graph = Graph()

    # first define input
    graph.add_input(name='stim', input_shape=input_shape)

    #first convolutional layer
    graph.add_node(TimeDistributedConvolution2D(num_filters[0], filter_size[0], filter_size[1], init=weight_init, border_mode='valid', subsample=(1, 1), W_regularizer=l2(reg)), name='conv0', input='stim')

    # Add relu activation
    graph.add_node(Activation('relu'), name='r0', input='conv0')

    # max pooling layer
    graph.add_node(TimeDistributedMaxPooling2D(pool_size=(2, 2)), name='mp', input='r0')

    #Add an rnn per subunit
    rnn_arr = []
    for i in range(num_filters[0]):
        graph.add_node(SubunitSlice(i), name='slice'+str(i), input='mp')
        graph.add_node(TimeDistributedFlatten(), name='flatten'+str(i), input='slice'+str(i))
        if rnn_flag == 2:
           graph.add_node(LSTM(num_hidden, return_sequences=True), name='rnn'+str(i), input='flatten'+str(i))
        elif rnn_flag == 1:
           graph.add_node(SimpleRNN(num_hidden, return_sequences=True), name='rnn'+str(i), input='flatten'+str(i))
        else: #default if not specified, applies 1 1x1 filter to each subunit type, each input will be of shape (nb_samples, time, 1, 19, 19) 
           graph.add_node(ConvRNN(filter_dim=(1, 1, 1), reshape_dim=(1, 19, 19), return_sequences=True), name='rnn'+str(i), input='flatten'+str(i))
        rnn_arr.append('rnn'+str(i))

    if rnn_flag == 1 or rnn_flag == 2:
       # Add first dense layer after concatenating RNN outputs
       graph.add_node(TimeDistributedDense(num_filters[1], init=weight_init, W_regularizer=l2(reg)), name='dense0', inputs=rnn_arr, merge_mode='concat', concat_axis=-1)
       # Add relu activation
       graph.add_node(Activation('relu'), name='r1', input='dense0')
    else:
       # Make network fully convolutional
       # First, we reshape RNN output to be (nb_samples, time, nb_subunits, rows, cols)
       graph.add_node(ImgReshape(num_filters[0]), name='reshape', inputs=rnn_arr, merge_mode='concat', concat_axis=-1)
       # Next, we apply convolutional layer to every timestep
       graph.add_node(TimeDistributedConvolution2D(num_filters[1], 9, 9, init=weight_init, border_mode='valid', subsample=(1, 1), W_regularizer=l2(reg)), name='finalunits', input='reshape')
       # Add relu activation
       graph.add_node(Activation('relu'), name='rect1', input='finalunits')
       graph.add_node(TimeDistributedFlatten(), name='r1', input='rect1')

    # Add a final dense (affine) layer, with a softplus at the end
    graph.add_node(TimeDistributedDense(nout, init=weight_init, W_regularizer=l2(reg), activation='softplus'), name='dense1', input='r1')
    
    graph.add_output(name='loss', input='dense1')

    with notify('Compiling'):
        graph.compile(optimizer, {'loss': loss})
    return graph
'''
    # Add first dense layer after concatenating RNN outputs
    graph.add_node(Dense(num_filters[1], init=weight_init, W_regularizer=l2(reg)), name='dense0', inputs=rnn_arr, merge_mode='concat')

    # Add relu activation
    graph.add_node(Activation('relu'), name='r1', input='dense0')
    
    # Add a final dense (affine) layer
    graph.add_node(Dense(nout, init=weight_init, W_regularizer=l2(reg)), name='dense1', input='r1')
    
    # Finish it off with a parametrized softplus
    graph.add_node(ParametricSoftplus(), name='softplus', input='dense1')
    
    graph.add_output(name='loss', input='softplus')
'''

def generalizedconvnet(input_shape, nout,
                       architecture=('conv', 'relu', 'pool', 'flatten', 'affine', 'relu', 'affine'),
                       num_filters=(4, -1, -1, -1, 16),
                       filter_sizes=(9, -1, -1, -1, -1),
                       weight_init='normal',
                       dropout=0.0,
                       dropout_type='binary',
                       l2_reg=0.0,
                       sigma=0.01):
    """Generic convolutional neural network

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

    num_filters : tuple, optional
        Number of filters in each layer. Default: [4, 16]

    filter_sizes : tuple, optional
        Convolutional filter size. Default: [9]
        Assumes that the filter is square.

    loss : string or object, optional
        A Keras objective. Default: 'poisson_loss'

    optimizer : string or object, optional
        A Keras optimizer. Default: 'adam'

    weight_init : string
        weight initialization. Default: 'normal'

    l2_reg : float, optional
        How much l2 regularization to apply to all filter weights

    """
    layers = list()

    for layer_id, layer_type in enumerate(architecture):

        # convolutional layer
        if layer_type == 'conv':
            if layer_id == 0:
                # initial convolutional layer
                layers.append(Convolution2D(num_filters[0], filter_sizes[0], filter_sizes[0],
                                            input_shape=input_shape, init=weight_init,
                                            border_mode='same', subsample=(1, 1), W_regularizer=l2(l2_reg)))
            else:
                layers.append(Convolution2D(num_filters[layer_id], filter_sizes[layer_id],
                                            filter_sizes[layer_id], init=weight_init, border_mode='same',
                                            subsample=(1, 1), W_regularizer=l2(l2_reg)))

        # Add relu activation
        if layer_type == 'relu':
            layers.append(Activation('relu'))

        # Add requ activation
        if layer_type == 'requ':
            layers.append(Activation('requ'))

        # Add exp activation
        if layer_type == 'exp':
            layers.append(Activation('exp'))

        # max pooling layer
        if layer_type =='pool':
            layers.append(MaxPooling2D(pool_size=(2, 2)))

        # flatten
        if layer_type == 'flatten':
            layers.append(Flatten())

        # dropout
        if layer_type == 'dropout':
            if dropout_type == 'gaussian':
                layers.append(GaussianDropout(dropout))
            else:
                layers.append(Dropout(dropout))

        # batch normalization
        if layer_type == 'batchnorm':
            layers.append(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None))

        # rnn
        if layer_type == 'rnn':
            num_hidden = 100
            layers.append(SimpleRNN(num_hidden, return_sequences=False, go_backwards=False, 
                        init='glorot_uniform', inner_init='orthogonal', activation='tanh',
                        W_regularizer=l2(l2_reg), U_regularizer=l2(l2_reg), dropout_W=0.1,
                        dropout_U=0.1))

        # noise layer
        if layer_type == 'noise':
            layers.append(GaussianNoise(sigma))

        # Add dense (affine) layer
        if layer_type == 'affine':
            if layer_id == len(architecture) - 1:
                # add final affine layer with softplus activation
                layers.append(Dense(nout, init=weight_init,
                                    W_regularizer=l2(l2_reg),
                                    activation='softplus'))
            else:
                layers.append(Dense(num_filters[layer_id], init=weight_init, W_regularizer=l2(l2_reg)))

    return layers
