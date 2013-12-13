#!/usr/bin/python2
#-*- coding: utf-8 -*-

import numpy
import theano
import theano.tensor as T

from reid.models.block import Block
from reid.utils import numpy_rng


class FullyConnectedLayer(Block):
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
    def __init__(self, input_shape, filter_shape, pool_shape, active_func=None):
        self._input_shape = input_shape
        self._filter_shape = filter_shape
        self._pool_shape = pool_shape
        self._active_func = active_func

        init_W = numpy.asarray(numpy_rng.uniform(
            low=-numpy.sqrt(3.0 / numpy.prod(filter_shape[1:])),
            high=numpy.sqrt(3.0 / numpy.prod(filter_shape[1:])),
            size=filter_shape), dtype=theano.config.floatX)

        self._W = theano.shared(value=init_W, name='W', borrow=True)

        init_b = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)

        self._b = theano.shared(value=init_b, name='b', borrow=True)

        self._params = [self._W, self._b]

    def get_output(self, x):
        z = T.nnet.conv.conv2d(
            input=x.reshape(self._input_shape),
            filters=self._W,
            filter_shape=self._filter_shape)

        y = T.signal.downsample.max_pool_2d(
            input=z,
            ds=self._pool_shape,
            ignore_border=True)

        y = y + self._b.dimshuffle('x', 0, 'x', 'x')

        y = y.ravel()

        return y if self._active_func is None else self._active_func(y)

