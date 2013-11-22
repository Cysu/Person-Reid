#!/usr/bin/python2
# -*- coding: utf-8 -*-

import theano.tensor as T


def MeanBinaryCrossEntropy(output, target):
    return T.nnet.binary_crossentropy(output, target).sum(axis=1).mean()

def MeanSquareError(output, target):
    return ((output - target) ** 2).sum(axis=1).mean()