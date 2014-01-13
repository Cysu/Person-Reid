#!/usr/bin/python2
# -*- coding: utf-8 -*-

import theano.tensor as T

# lambda functions cannot be pickled

def identity(x):
    return x

def rectifier(x):
    return x * (x > 0.0)

sigmoid = T.nnet.sigmoid

softmax = T.nnet.softmax

tanh = T.tanh
