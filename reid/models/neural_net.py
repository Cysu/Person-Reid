#!/usr/bin/python2
# -*- coding: utf-8 -*-

import theano.tensor as T


class NeuralNet(object):
    """Base class of neural network (NeuralNet)"""

    def __init__(self, cost_func=None, error_func=None):

        self._cost_func = cost_func
        self._error_func = error_func

        raise NotImplementedError(
            str(type(self)) + " does not implement __init__")

    def get_outputs(self, x):

        raise NotImplementedError(
            str(type(self)) + " does not implement get_outputs")

    def get_cost_updates(self, x, target, learning_rate):

        assert self._cost_func is not None

        y = self.get_outputs(x)[-1]

        cost = self._cost_func(output=y, target=target)

        grads = T.grad(cost, self.params)

        updates = []
        for p, g in zip(self.params, grads):
            updates.append((p, p - learning_rate * g))

        return (cost, updates)
        
    def get_error(self, x, target):

        assert self._error_func is not None

        y = self.get_outputs(x)[-1]

        error = self._error_func(output=y, target=target)

        return error