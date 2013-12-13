#!/usr/bin/python2
# -*- coding: utf-8 -*-


class Block(object):
    """Abstract base class for neural network layers and models"""

    def __init__(self):
        self._params = None

    @property
    def parameters(self):
        return self._params

    @parameters.setter
    def parameters(self, value):
        self._params = value

    def get_output(self, x):
        raise NotImplementedError