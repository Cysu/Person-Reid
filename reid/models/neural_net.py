#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy
import theano
import theano.tensor as T

from reid.models.block import Block
from reid.models.layers import FullConnLayer


def get_cost_updates(model, cost_func, x, target, learning_rate, regularize=0, momentum=0):
    y = model.get_output(x)
    cost = cost_func(output=y, target=target)
    if regularize > 0:
        cost += regularize * model.get_regularization(2)  # Use 2-norm by default

    grads = T.grad(cost, model.parameters)

    if momentum == 0:
        updates = [(p, p-learning_rate*g) for p, g in zip(model.parameters, grads)]
    else:
        prev_grads = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
            dtype=p.dtype), borrow=True) for p in model.parameters]

        updates = []
        for param, grad, prev_grad in zip(model.parameters, grads, prev_grads):
            updates.append((param, param - learning_rate*(grad + momentum*prev_grad)))
            updates.append((prev_grad, grad))

    return (cost, updates)

def get_error(model, error_func, x, target):
    y = model.get_output(x)
    return error_func(output=y, target=target)


class NeuralNet(Block):
    """A composition of several blocks"""

    def __init__(self, blocks):
        super(NeuralNet, self).__init__()

        self._blocks = blocks

        self._params = []
        for block in blocks:
            self._params.extend(block.parameters)

    def get_output(self, x):
        for block in self._blocks:
            y = block.get_output(x)
            x = y
        return y

    def get_regularization(self, l):
        return sum([block.get_regularization(l) for block in self._blocks])

    
class AutoEncoder(NeuralNet):
    """Class for auto-encoder

    The auto-encoder has symmetric structure. Each layer is fully connected.
    """

    def __init__(self, layer_sizes, active_funcs):
        """Initialize the auto-encoder

        Args:
            layer_sizes: A list of integers. The first one is the input size.
            The last one is the middlest layer's size.

            active_funcs: A list of layer-wise active functions.
        """
        super(AutoEncoder, self).__init__()

        n_layers = len(active_funcs)
        assert n_layers == len(layer_sizes) - 1

        self._blocks, self._params = [], []

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

    def __init__(self, blocks):
        """Initialize the multiway neural network

        Args:
            blocks: A list of parallel typical neural networks
        """

        super(MultiwayNeuralNet, self).__init__()

        self._blocks = blocks

    def get_output(self, x):
        """Get the list of output data

        Args:
            x: A list of theano symbols each representing an input to a column

        Returns:
            A list of output data
        """

        return [block.get_output(data) for data, block in zip(x, self._blocks)]

    def get_regularization(self, l):
        return sum([block.get_regularization(l) for block in self._blocks])
