#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy
import theano
import theano.tensor as T

from neural_net import NeuralNet


class Layer(NeuralNet):
    
    def __init__(self, numpy_rng, input_size, output_size,
                 active_func=None, cost_func=None, error_func=None):

        self._active_func = active_func
        self._cost_func = cost_func
        self._error_func = error_func

        init_W = numpy.asarray(numpy_rng.uniform(
            low=-4 * numpy.sqrt(6.0 / (input_size + output_size)),
            high=4 * numpy.sqrt(6.0 / (input_size + output_size)),
            size=(input_size, output_size)), dtype=theano.config.floatX)

        self.W = theano.shared(value=init_W, name='W', borrow=True)

        init_b = numpy.zeros(output_size, dtype=theano.config.floatX)

        self.b = theano.shared(value=init_b, name='b', borrow=True)

        self.params = [self.W, self.b]

    def get_outputs(self, x):

        z = T.dot(x, self.W) + self.b

        return [z] if self._active_func is None else [self._active_func(z)]
