#!/usr/bin/python2
# -*- coding: utf-8 -*-

import theano.tensor as T


class MultiLayerPerceptron(object):

    def __init__(self, layers):

        self._layers = layers

        self.params = []
        for layer in layers:
            self.params.extend(layer.params)

    def get_outputs(self, x):

        outputs = []

        for layer in self._layers:
            x = layer.get_outputs(x)
            outputs.append(x)

        return outputs

    def get_cost(self, x, target):

        y = self.get_outputs(x)[-1]

        # cost = T.mean(T.sqrt(T.sum((y - target) ** 2, axis=1)))
        cost = T.nnet.binary_crossentropy(y, target).mean()

        return cost

    def get_updates(self, cost, learning_rate):

        grads = T.grad(cost, self.params)

        updates = []
        for p, g in zip(self.params, grads):
            updates.append((p, p - learning_rate * g))

        return updates

    def get_error(self, x, target):

        y = self.get_outputs(x)[-1]

        error = T.mean(T.sqrt(T.sum((y - target) ** 2, axis=1)))

        return error