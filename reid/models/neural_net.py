#!/usr/bin/python2
# -*- coding: utf-8 -*-

from reid.models.block import Block
from reid.models.layers import FullConnLayer


class NeuralNet(Block):
    """Composite blocks in a sequential manner"""

    def __init__(self, blocks, const_params=None):
        """Initialize the neural network

        Args:
            blocks: A list of sub networks
            const_params: A list of boolean values indicating whether the
                parameters of corresponding sub network is constant, i.e.,
                the parameters will not be updated when doing gradient
                descent. None if all of them are not constant.
        """

        super(NeuralNet, self).__init__()

        self._blocks = blocks

        if const_params is None:
            const_params = [False] * len(blocks)

        self._params = []
        for block, is_const in zip(blocks, const_params):
            if not is_const:
                self._params.extend(block.parameters)

    def get_output(self, x):
        """Get the final result of passing input data to each sub network
        sequentially

        Args:
            x: A theano matrix with each row being a feature vector
        """

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


class MultiwayNeuralNet(NeuralNet):
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

        super(MultiwayNeuralNet, self).__init__(blocks)

    def get_output(self, x):
        """Get the list of output data

        Args:
            x: A list of theano symbols each representing an input to a column

        Returns:
            A list of output data
        """

        return [block.get_output(data) for data, block in zip(x, self._blocks)]
