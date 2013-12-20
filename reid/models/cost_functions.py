#!/usr/bin/python2
# -*- coding: utf-8 -*-

import theano.tensor as T


def mean_binary_cross_entropy(output, target):
    return T.nnet.binary_crossentropy(output, target).sum(axis=1).mean()

def mean_square_error(output, target):
    return ((output - target) ** 2).sum(axis=1).mean()
