#!/usr/bin/python2
# -*- coding: utf-8 -*-

import theano.tensor as T

from reid.models.block import Block


class NeuralNet(Block):
    """A composition of several blocks"""

    def __init__(self, blocks, cost_func=None, error_func=None):
        self._blocks = blocks
        self._cost_func = cost_func
        self._error_func = error_func

        self._params = []
        for block in blocks:
            self._params.extend(block.parameters)

    def get_output(self, x):
        for block in self._blocks:
            y = block.get_output(x)
            x = y
        return y

    def get_cost_updates(self, x, target, learning_rate):
        assert self._cost_func is not None

        y = self.get_output(x)
        cost = self._cost_func(output=y, target=target)
        grads = T.grad(cost, self._params)

        updates = []
        for p, g in zip(self._params, grads):
            updates.append((p, p - learning_rate * g))

        return (cost, updates)
        
    def get_error(self, x, target):
        assert self._error_func is not None

        y = self.get_output(x)
        error = self._error_func(output=y, target=target)

        return error
