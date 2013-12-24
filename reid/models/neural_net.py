#!/usr/bin/python2
# -*- coding: utf-8 -*-

import theano.tensor as T

from reid.models.block import Block
from reid.models.layers import FullConnLayer


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
        y = self.get_output(x)
        cost = self._cost_func(output=y, target=target)
        grads = T.grad(cost, self._params)

        updates = []
        for p, g in zip(self._params, grads):
            updates.append((p, p - learning_rate * g))

        return (cost, updates)
        
    def get_error(self, x, target):
        y = self.get_output(x)
        error = self._error_func(output=y, target=target)

        return error


class AutoEncoder(NeuralNet):
    """Class for auto-encoder

    The auto-encoder has symmetric structure. Each layer is fully connected.
    """

    def __init__(self, layer_sizes, active_funcs,
                 cost_func=None, error_func=None):
        """Initialize the auto-encoder

        Args:
            layer_sizes: A list of integers. The first one is the input size.
            The last one is the middlest layer's size.

            active_funcs: A list of layer-wise active functions.
        """

        n_layers = len(active_funcs)

        assert n_layers == len(layer_sizes) - 1

        self._cost_func = cost_func
        self._error_func = error_func
        self._blocks = []

        self._params = []

        # Build feature extraction layers
        for i in xrange(n_layers):
            layer = FullConnLayer(input_size=layer_sizes[i],
                                  output_size=layer_sizes[i+1],
                                  active_func=active_funcs[i])
            self._blocks.append(layer)
            self._params.extend(layer.parameters)

        # Build reconstruction layers
        for i in xrange(n_layers-1, -1, -1):
            layer = FullConnLayer(input_size=layer_sizes[i+1],
                                  output_size=layer_sizes[i],
                                  active_func=active_funcs[i],
                                  W=self._blocks[i].parameters[0].T)
            self._blocks.append(layer)
            self._params.append(layer.parameters[1])


class MultiwayNeuralNet(Block):
    """Multiway neural network

    The multiway neural network consists of several parallel typical neural 
    networks. It receives a list of input data. Each is passed through a typcial
    neural network and thus forms the list of output data.
    """

    def __init__(self, blocks, cost_func=None, error_func=None):
        """Initialize the multiway neural network

        Args:
            blocks: A list of parallel typical neural networks
        """

        self._blocks = blocks
        self._cost_func = cost_func
        self._error_func = error_func

    def get_output(self, x):
        """Get the list of output data

        Args:
            x: A list or numpy ndarray. Each item should be a numpy ndarray 
                representing an input data.

        Returns:
            A list of output data
        """

        ret = []
        for data, block in zip(x, self._blocks):
            ret.append(block.get_output(data))

        return ret
