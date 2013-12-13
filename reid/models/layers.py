#!/usr/bin/python2
#-*- coding: utf-8 -*-

import numpy
import theano
import theano.tensor as T

from reid.models.block import Block
from reid.utils import numpy_rng


class FullConnLayer(Block):
    def __init__(self, input_size, output_size, active_func=None):
        self._active_func = active_func

        init_W = numpy.asarray(numpy_rng.uniform(
            low=-4 * numpy.sqrt(6.0 / (input_size + output_size)),
            high=4 * numpy.sqrt(6.0 / (input_size + output_size)),
            size=(input_size, output_size)), dtype=theano.config.floatX)

        self._W = theano.shared(value=init_W, name='W', borrow=True)

        init_b = numpy.zeros(output_size, dtype=theano.config.floatX)

        self._b = theano.shared(value=init_b, name='b', borrow=True)

        self._params = [self._W, self._b]

    def get_output(self, x):
        z = T.dot(x, self._W) + self._b
        return z if self._active_func is None else self._active_func(z)


class ConvPoolLayer(Block):
    def __init__(self, filter_shape, pool_shape,
                 image_shape=None, active_func=None, flatten_output=False):
        self._filter_shape = filter_shape
        self._pool_shape = pool_shape
        self._image_shape = image_shape
        self._active_func = active_func
        self._flatten_output = flatten_output

        init_W = numpy.asarray(numpy_rng.uniform(
            low=-numpy.sqrt(3.0 / numpy.prod(filter_shape[1:])),
            high=numpy.sqrt(3.0 / numpy.prod(filter_shape[1:])),
            size=filter_shape), dtype=theano.config.floatX)

        self._W = theano.shared(value=init_W, name='W', borrow=True)

        init_b = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)

        self._b = theano.shared(value=init_b, name='b', borrow=True)

        self._params = [self._W, self._b]

    def get_output(self, x):
        if self._image_shape is not None:
            input_shape = (x.shape[0],) + self._image_shape
            x = x.reshape(input_shape)

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

        if self._flatten_output:
            y = y.reshape((y.shape[0], y.shape[1:].prod()))

        return y
