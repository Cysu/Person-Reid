#!/usr/bin/python2
# -*- coding: utf-8 -*-

from reid.models.neural_net import NeuralNet


class MultiLayerPerceptron(NeuralNet):

    def __init__(self, layers, cost_func, error_func):
        self._layers = layers
        self._cost_func = cost_func
        self._error_func = error_func

        self._params = []
        for layer in layers:
            self._params.extend(layer._params)

    def get_layers(self):
        return self._layers

    def get_outputs(self, x):
        outputs = []
        for layer in self._layers:
            y = layer.get_outputs(x)
            outputs.extend(y)
            x = y[-1]

        return outputs
