#!/usr/bin/python2
# -*- coding: utf-8 -*-

import theano.tensor as T


# TODO: rectifier active function cannot be pickled

identity = lambda x: x

rectifier = lambda x: x * (x > 0.0)

sigmoid = T.nnet.sigmoid

tanh = T.tanh