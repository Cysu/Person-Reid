#!/usr/bin/python2
# -*- coding: utf-8 -*-

from neural_net import NeuralNet


class MultiLayerPerceptron(NeuralNet):

    def __init__(self, layers, cost_func, error_func):

        self._layers = layers
        self._cost_func = cost_func
        self._error_func = error_func

        self.params = []
        for layer in layers:
            self.params.extend(layer.params)

    def get_outputs(self, x):

        outputs = []

        for layer in self._layers:
            x = layer.get_outputs(x)
            outputs.append(x)

        return outputs