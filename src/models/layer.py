#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy
import theano
import theano.tensor as T


class Layer(object):
    
    def __init__(self, numpy_rng, input_size, output_size, active_func):

        self.f = active_func

        init_W = numpy.asarray(numpy_rng.uniform(
            low=-4 * numpy.sqrt(6.0 / (input_size + output_size)),
            high=4 * numpy.sqrt(6.0 / (input_size + output_size)),
            size=(input_size, output_size),
            dtype=theano.config.floatX))

        self.W = theano.shared(value=init_W, name='W', borrow=True)

        init_b = numpy.zeros(output_size, dtype=theano.config.floatX)

        self.b = theano.shared(value=init_b, name='b', borrow=True)

        self.params = [self.W, self.b]

    def get_outputs(self, x):

        return self.f(T.dot(x, self.W) + self.b)

    def get_cost(self, x, target):

        y = self.get_outputs(x)

        cost = T.mean(T.sqrt(T.sum((y - target) ** 2, axis=1)))

        return cost

    def get_updates(self, cost, learning_rate):

        grads = T.grad(cost, self.params)

        updates = []
        for p, g in zip(self.params, grads):
            updates.append((p, p - learning_rate * g))

        return updates

    def get_error(self, x, target):

        y = self.get_outputs(x)

        error = T.mean(T.sqrt(T.sum((y - target) ** 2, axis=1)))

        return error
