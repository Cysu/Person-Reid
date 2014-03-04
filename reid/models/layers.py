#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy
import theano
import theano.tensor as T
from reid.models.block import Block
from reid.models import active_functions as actfuncs
from reid.utils.math_utils import numpy_rng


class FullConnLayer(Block):
    """Fully connected layer"""

    def __init__(self, input_size, output_size, active_func=None,
                 W=None, b=None,
                 through=False):
        super(FullConnLayer, self).__init__()

        self._active_func = active_func
        self._through = through

        if W is None:
            W_bound = numpy.sqrt(6.0 / (input_size + output_size))

            if active_func == actfuncs.sigmoid: W_bound *= 4

            init_W = numpy.asarray(numpy_rng.uniform(
                low=-W_bound, high=W_bound,
                size=(input_size, output_size)), dtype=theano.config.floatX)

            self._W = theano.shared(value=init_W, borrow=True)
        else:
            self._W = W

        if b is None:
            init_b = numpy.zeros(output_size, dtype=theano.config.floatX)

            self._b = theano.shared(value=init_b, borrow=True)
        else:
            self._b = b

        self._params = [self._W, self._b]

    def get_output(self, x):
        z = T.dot(x, self._W) + self._b
        z = z if self._active_func is None else self._active_func(z)
        return (z, []) if not self._through else (z, [z])

    def get_regularization(self, l):
        return self._W.norm(l)


class ConvPoolLayer(Block):
    """Convolutional and max-pooling layer"""

    def __init__(self, filter_shape, pool_shape,
                 image_shape=None, active_func=None,
                 flatten=False, through=False):
        """Initialize the convolutional and max-pooling layer

        Args:
            filter_shape: 4D-tensor, (n_filters, n_channels, n_rows, n_cols)
            pool_shape: 2D-tensor, (n_rows, n_cols)
            image_shape: None if the input is always 4D-tensor, (n_images,
                n_channels, n_rows, n_cols). If the input images are represented
                as vectors, then a 3D-tensor, (n_channels, n_rows, n_cols) is
                required.
            active_func: Active function of this layer
            flatten: True if the output image should be flattened as a vector
            through: True if the output should be passed through
        """

        super(ConvPoolLayer, self).__init__()

        self._filter_shape = filter_shape
        self._pool_shape = pool_shape
        self._image_shape = image_shape
        self._active_func = active_func
        self._flatten = flatten
        self._through = through

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(pool_shape)

        W_bound = numpy.sqrt(6.0 / (fan_in + fan_out))

        if active_func == actfuncs.sigmoid: W_bound *= 4

        init_W = numpy.asarray(numpy_rng.uniform(
            low=-W_bound, high=W_bound,
            size=filter_shape), dtype=theano.config.floatX)

        self._W = theano.shared(value=init_W, borrow=True)

        init_b = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)

        self._b = theano.shared(value=init_b, borrow=True)

        self._params = [self._W, self._b]

    def get_output(self, x):
        if self._image_shape is not None:
            x = x.reshape((x.shape[0],) + self._image_shape)

        z = T.nnet.conv.conv2d(
            input=x,
            filters=self._W,
            filter_shape=self._filter_shape)

        y = T.signal.downsample.max_pool_2d(
            input=z,
            ds=self._pool_shape,
            ignore_border=True)

        y = y + self._b.dimshuffle('x', 0, 'x', 'x')

        if self._active_func is not None:
            y = self._active_func(y)

        if self._flatten:
            y = y.flatten(2)

        return (y, []) if not self._through else (y, [y])

    def get_regulariation(self, l):
        return self._W.norm(l)


class FilterParingLayer(Block):
    """Filter paring layer

    """

    def __init__(self, image_shape, maxout_grouping=None,
                 flatten=False, through=False):
        self._image_shape = image_shape
        self._maxout_grouping = maxout_grouping
        self._flatten = flatten
        self._through = through

    def get_output(self, xa, xb):
        n_samples = xa.shape[0]
        n_channels, h, w = self._image_shape

        xa = xa.reshape(n_samples*n_channels*h, w)
        xb = xb.reshape(n_samples*n_channels*h, w)

        y = [xa[:, i].dimshuffle(0, 'x') * xb for i in xrange(w)]
        y = T.concatenate(y, axis=1)

        if self._maxout_grouping is None:
            y = y.reshape([n_samples, n_channels, h, w, w]).max(axis=1)
        else:
            n_groups = self._maxout_grouping
            y = y.reshape([n_samples, n_groups, n_channels/n_groups, h, w])
            y = y.max(axis=1).reshape([n_samples, n_channels/n_groups*h, w, w])

        if self._flatten:
            y = y.flatten(2)

        return (y, []) if not self._through else (y, [y])


class CompLayer(Block):
    """Composition layer

    Composition layer concatenates a list of theano tensors into a theano
    matrix. The tensor should have at least two dimensions. Along the first
    dimension are data samples.
    """

    def __init__(self, strategy=None, through=False):
        """Initialize the composition layer

        Args:
            strategy: None to just concatenate the list of theano tensors
                together. "Maxout" to perform max-out grouping on the list
                of theano tensors.
        """

        super(CompLayer, self).__init__()

        self._strategy = strategy
        self._through = through

    def get_output(self, x):
        """Get the concatenated matrix

        Args:
            x: A list of theano tensors

        Returns:
            The concatenated matrix
        """

        z = [t.flatten(2) for t in x]

        if self._strategy == 'Maxout':
            m, n, g = z[0].shape[0], z[0].shape[1], len(z)
            y = T.concatenate(z, axis=0).reshape((g, m, n)).max(axis=0)
        else:
            y = T.concatenate(z, axis=1)

        return (y, []) if not self._through else (y, [y])

class DecompLayer(Block):
    """Decomposition layer

    Decomposition layer separates a theano matrix into a list of theano tensors.
    The tensor has at least two dimensions. Along the first dimension are data
    samples.
    """

    def __init__(self, data_shapes, active_funcs=None, through=False):
        """Initialize the decomposition layer

        Args:
            data_shapes: A list of tuples representing shapes of each data
                sample
            active_funcs: A list of active functions for each separated theano
                tensor
            through: True if the output should be passed through
        """

        super(DecompLayer, self).__init__()

        self._data_shapes = data_shapes
        self._active_funcs = active_funcs
        self._through = through

        self._segments = [0]
        for i, shape in enumerate(data_shapes):
            self._segments.append(self._segments[i] + numpy.prod(shape))

    def get_output(self, x):
        """Get the separated list of theano tensors with specified activation

        Args:
            x: A concatenated theano matrix

        Returns:
            The list of theano tensors with specified activation
        """

        m = x.shape[0]
        ret = [0] * len(self._data_shapes)

        for i, shape in enumerate(self._data_shapes):
            l, r = self._segments[i], self._segments[i+1]
            ret[i] = x[:, l:r].reshape((m,) + shape)
            if self._active_funcs is not None and self._active_funcs[i] is not None:
                ret[i] = self._active_funcs[i](ret[i])

        return (ret, []) if not self._through else (ret, [ret])


class CloneLayer(Block):
    """Clone layer

    Clone layer clones the input tensor or list. All cloned instances share
    same reference. Typically, the layer is followed by a MultiwayNeuralNet.
    """

    def __init__(self, n_copies, through=False):
        """Initialize the clone layer

        Args:
            n_copies: The number of copies to be cloned
            through: True if the output should be passed through
        """

        self._n_copies = n_copies
        self._through = through


    def get_output(self, x):
        """Get the cloned list of tensors or lists

        Args:
            x: A theano tensor or python list

        Returns:
            A list of [x] * n_copies
        """

        ret = [x] * self._n_copies

        return (ret, []) if not self._through else (ret, [ret])
