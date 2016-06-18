from keras import backend as K
from keras import activations, initializations, regularizers
from keras.layers.core import Masking
from keras.engine.topology import Layer
from keras.engine import InputSpec

from keras.layers.convolutional import conv_output_length
import numpy as np


class RecurrentConv2D(Layer):
    '''Abstract base class for recurrent layers.
    Do not use in a model -- it's not a functional layer!

    All recurrent layers (GRU, LSTM, SimpleRNN) also
    follow the specifications of this class and accept
    the keyword arguments listed below.

    # Input shape
        5D tensor with shape `(nb_samples, timesteps, channels, rows, cols)`.

    # Output shape
        - if `return_sequences`: 5D tensor with shape
            `(nb_samples, timesteps, channels,rows,cols)`.
        - else, 2D tensor with shape `(nb_samples, channels,rows,cols)`.

    # Arguments
        weights: list of numpy arrays to set as initial weights.
            The list should have 3 elements, of shapes:
            `[(input_dim, nb_filter), (nb_filter, nb_filter), (nb_filter,)]`.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        go_backwards: Boolean (default False).
            If True, rocess the input sequence backwards.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        nb_filter: Number of convolution filters to use.
        nb_row: Number of rows in the convolution kernel.
        nb_col: Number of columns in the convolution kernel.
            is required when using this layer as the first layer in a model.
        input_shape: input_shape

    # Masking
        This layer supports masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
        set to `True`.
        **Note:** for the time being, masking is only supported with Theano.

    # TensorFlow warning
        For the time being, when using the TensorFlow backend,
        the number of timesteps used must be specified in your model.
        Make sure to pass an `input_length` int argument to your
        recurrent layer (if it comes first in your model),
        or to pass a complete `input_shape` argument to the first layer
        in your model otherwise.


    # Note on using statefulness in RNNs
        You can set RNN layers to be 'stateful', which means that the states
        computed for the samples in one batch will be reused as initial states
        for the samples in the next batch.
        This assumes a one-to-one mapping between
        samples in different successive batches.

        To enable statefulness:
            - specify `stateful=True` in the layer constructor.
            - specify a fixed batch size for your model, by passing
                a `batch_input_size=(...)` to the first layer in your model.
                This is the expected shape of your inputs *including the batch
                size*.
                It should be a tuple of integers, e.g. `(32, 10, 100)`.

        To reset the states of your model, call `.reset_states()` on either
        a specific layer, or on your entire model.
    '''
    input_ndim = 5

    def __init__(self, weights=None,
                 return_sequences=False, go_backwards=False, stateful=False,
                 nb_row=None, nb_col=None, nb_filter=None,
                 dim_ordering=None,
                 input_dim=None, input_length=None, **kwargs):
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.go_backwards = go_backwards
        self.stateful = stateful

        self.nb_row = nb_row
        self.nb_col = nb_col
        self.nb_filter = nb_filter
        self.dim_ordering = dim_ordering

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)

        super(RecurrentConv2D, self).__init__(**kwargs)

    def compute_mask(self, input, mask):
        if self.return_sequences:
            return mask
        else:
            return None

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            rows = input_shape[2+1]
            cols = input_shape[3+1]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1+1]
            cols = input_shape[2+1]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        rows = conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])

        if self.return_sequences:
            if self.dim_ordering == 'th':
                return (input_shape[0], input_shape[1],
                        self.nb_filter, rows, cols)
            elif self.dim_ordering == 'tf':
                return (input_shape[0], input_shape[1],
                        rows, cols, self.nb_filter)
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        else:
            if self.dim_ordering == 'th':
                return (input_shape[0], self.nb_filter, rows, cols)
            elif self.dim_ordering == 'tf':
                return (input_shape[0], rows, cols, self.nb_filter)
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def step(self, x, states):
        raise NotImplementedError

    def get_constants(self, x):
        return None

    def get_initial_states(self, X):
        # (samples, timesteps, row, col, filter)
        initial_state = K.zeros_like(X)
        # (samples,row, col, filter)
        initial_state = K.sum(initial_state, axis=1)
        # initial_state = initial_state[::,]
        initial_state = self.conv_step(initial_state, K.zeros(self.W_shape),
                                       border_mode=self.border_mode)

        initial_states = [initial_state for _ in range(2)]
        return initial_states

    def call(self, x, mask=None):
        constants = self.get_constants(x)

        assert K.ndim(x) == 5
        if K._BACKEND == 'tensorflow':
            if not self.input_shape[1]:
                raise Exception('When using TensorFlow, you should define ' +
                                'explicitely the number of timesteps of ' +
                                'your sequences. Make sure the first layer ' +
                                'has a "batch_input_shape" argument ' +
                                'including the samples axis.')

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)

        last_output, outputs, states = K.rnn(self.step, x,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants)
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "return_sequences": self.return_sequences,
                  "go_backwards": self.go_backwards,
                  "stateful": self.stateful}
        if self.stateful:
            config['batch_input_shape'] = self.input_shape
        else:
            config['input_dim'] = self.input_dim
            config['input_length'] = self.input_length

        base_config = super(RecurrentConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LSTMConv2D(RecurrentConv2D):
    '''
    # Input shape
            5D tensor with shape:
            `(samples,time, channels, rows, cols)` if dim_ordering='th'
            or 5D tensor with shape:
            `(samples,time, rows, cols, channels)` if dim_ordering='tf'.

     # Output shape
        if return_sequences=False

            4D tensor with shape:
            `(samples, nb_filter, o_row, o_col)` if dim_ordering='th'
            or 4D tensor with shape:
            `(samples, o_row, o_col, nb_filter)` if dim_ordering='tf'.
        if return_sequences=True
            5D tensor with shape:
            `(samples, time,nb_filter, o_row, o_col)` if dim_ordering='th'
            or 5D tensor with shape:
            `(samples, time, o_row, o_col, nb_filter)` if dim_ordering='tf'.

        where o_row and o_col depend on the shape of the filter and
        the border_mode

        # Arguments
            nb_filter: Number of convolution filters to use.
            nb_row: Number of rows in the convolution kernel.
            nb_col: Number of columns in the convolution kernel.
            border_mode: 'valid' or 'same'.
            sub_sample: tuple of length 2. Factor by which to subsample output.
            Also called strides elsewhere.
            dim_ordering: "tf" if the feature are at the last dimension or "th"
            stateful : has not been checked yet.


            init: weight initialization function.
                Can be the name of an existing function (str),
                or a Theano function
                (see: [initializations](../initializations.md)).
            inner_init: initialization function of the inner cells.
            forget_bias_init: initialization function for the bias of the
            forget gate.
                [Jozefowicz et al.]
                (http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
                recommend initializing with ones.
            activation: activation function.
                Can be the name of an existing function (str),
                or a Theano function (see: [activations](../activations.md)).
            inner_activation: activation function for the inner cells.

    # References
        - [Convolutional LSTM Network: A Machine Learning Approach for
        Precipitation Nowcasting](http://arxiv.org/pdf/1506.04214v1.pdf)
        The current implementation does not include the feedback loop on the
        cells output
    '''
    def __init__(self, nb_filter, nb_row, nb_col,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid', dim_ordering="tf",
                 border_mode="valid", sub_sample=(1, 1),
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.border_mode = border_mode
        self.subsample = sub_sample

        assert dim_ordering in {'tf', "th"}, 'dim_ordering must be in {tf,"th}'
        self.dim_ordering = dim_ordering

        kwargs["nb_filter"] = nb_filter
        kwargs["nb_row"] = nb_row
        kwargs["nb_col"] = nb_col
        kwargs["dim_ordering"] = dim_ordering

        self.W_regularizer = W_regularizer
        self.U_regularizer = U_regularizer
        self.b_regularizer = b_regularizer
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        super(LSTMConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        #self.input = K.placeholder(input_shape)
        self.input_spec = [InputSpec(shape=input_shape)]

        if self.dim_ordering == 'th':
            stack_size = input_shape[1+1]
            self.W_shape = (self.nb_filter, stack_size,
                            self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            stack_size = input_shape[3+1]
            self.W_shape = (self.nb_row, self.nb_col,
                            stack_size, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        if self.dim_ordering == 'th':
            self.W_shape1 = (self.nb_filter, self.nb_filter,
                             self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            self.W_shape1 = (self.nb_row, self.nb_col,
                             self.nb_filter, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensor of shape (nb_filter)
            self.states = [None, None, None, None]

        self.W_i = self.init(self.W_shape)
        self.U_i = self.inner_init(self.W_shape1)
        self.b_i = K.zeros((self.nb_filter,))

        self.W_f = self.init(self.W_shape)
        self.U_f = self.inner_init(self.W_shape1)
        self.b_f = self.forget_bias_init((self.nb_filter,))

        self.W_c = self.init(self.W_shape)
        self.U_c = self.inner_init(self.W_shape1)
        self.b_c = K.zeros((self.nb_filter))

        self.W_o = self.init(self.W_shape)
        self.U_o = self.inner_init(self.W_shape1)
        self.b_o = K.zeros((self.nb_filter,))

        def append_regulariser(input_regulariser, param, regularizers_list):
            regulariser = regularizers.get(input_regulariser)
            if regulariser:
                regulariser.set_param(param)
                regularizers_list.append(regulariser)

        self.regularizers = []
        for W in [self.W_i, self.W_f, self.W_i, self.W_o]:
            append_regulariser(self.W_regularizer, W, self.regularizers)
        for U in [self.U_i, self.U_f, self.U_i, self.U_o]:
            append_regulariser(self.U_regularizer, U, self.regularizers)
        for b in [self.b_i, self.b_f, self.b_i, self.b_o]:
            append_regulariser(self.b_regularizer, b, self.regularizers)

        self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                  self.W_c, self.U_c, self.b_c,
                                  self.W_f, self.U_f, self.b_f,
                                  self.W_o, self.U_o, self.b_o]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided ' +
                            '(including batch size).')

        if self.return_sequences:
            out_row, out_col, out_filter = self.output_shape[2:]
        else:
            out_row, out_col, out_filter = self.output_shape[1:]

        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0],
                                  out_row, out_col, out_filter)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0],
                                  out_row, out_col, out_filter)))
        else:
            self.states = [K.zeros((input_shape[0],
                                    out_row, out_col, out_filter)),
                           K.zeros((input_shape[0],
                                    out_row, out_col, out_filter))]

    def conv_step(self, x, W, b=None, border_mode="valid"):
        input_shape = self.input_spec[0].shape

        conv_out = K.conv2d(x, W, strides=self.subsample,
                            border_mode=border_mode,
                            dim_ordering=self.dim_ordering,
                            image_shape=(input_shape[0],
                                         input_shape[2],
                                         input_shape[3],
                                         input_shape[4]),
                            filter_shape=self.W_shape)
        if b:
            if self.dim_ordering == 'th':
                conv_out = conv_out + K.reshape(b, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                conv_out = conv_out + K.reshape(b, (1, 1, 1, self.nb_filter))
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        return conv_out

    def conv_step_hidden(self, x, W, border_mode="valid"):
        # This new function was defined because the
        # image shape must be hardcoded
        input_shape = self.input_spec[0].shape
        output_shape = self.get_output_shape_for(input_shape)

        if self.return_sequences:
            out_row, out_col, out_filter = output_shape[2:]
        else:
            out_row, out_col, out_filter = output_shape[1:]

        conv_out = K.conv2d(x, W, strides=(1, 1),
                            border_mode=border_mode,
                            dim_ordering=self.dim_ordering,
                            image_shape=(input_shape[0],
                                         out_row, out_col,
                                         out_filter),
                            filter_shape=self.W_shape1)

        return conv_out

    def step(self, x, states):
        assert len(states) == 4
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_W = states[2]
        B_U = states[3]

        x_i = self.conv_step(x * B_W[0], self.W_i, self.b_i,
                             border_mode=self.border_mode)
        x_f = self.conv_step(x * B_W[1], self.W_f, self.b_f,
                             border_mode=self.border_mode)
        x_c = self.conv_step(x * B_W[2], self.W_c, self.b_c,
                             border_mode=self.border_mode)
        x_o = self.conv_step(x * B_W[3], self.W_o, self.b_o,
                             border_mode=self.border_mode)

        # U : from nb_filter to nb_filter
        # Same because must be stable in the ouptut space
        h_i = self.conv_step_hidden(h_tm1, self.U_i * B_U[0],
                                    border_mode="same")
        h_f = self.conv_step_hidden(h_tm1, self.U_f * B_U[1],
                                    border_mode="same")
        h_c = self.conv_step_hidden(h_tm1, self.U_c * B_U[2],
                                    border_mode="same")
        h_o = self.conv_step_hidden(h_tm1, self.U_o * B_U[3],
                                    border_mode="same")

        i = self.inner_activation(x_i + h_i)
        f = self.inner_activation(x_f + h_f)
        c = f * c_tm1 + i * self.activation(x_c + h_c)
        o = self.inner_activation(x_o + h_o)
        h = o * self.activation(c)

        return h, [h, c]
 
    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(4)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(4)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        return constants

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "nb_filter": self.nb_filter,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  'dim_ordering': self.dim_ordering,
                  'border_mode': self.border_mode,
                  "inner_activation": self.inner_activation.__name__}
        base_config = super(LSTMConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))