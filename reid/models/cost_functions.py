#!/usr/bin/python2
# -*- coding: utf-8 -*-

import theano.tensor as T


def mean_binary_cross_entropy(output, target):
    return T.nnet.binary_crossentropy(output, target).sum(axis=1).mean()

def mean_square_error(output, target):
    return ((output - target) ** 2).sum(axis=1).mean()

def mean_negative_loglikelihood(output, target):
    target = target.astype('int32').ravel()
    return -T.mean(T.log(output)[T.arange(target.shape[0]), target])

def mean_number_misclassified(output, target):
    pred = T.argmax(output, axis=1, keepdims=True)
    return T.neq(pred, target).mean()

def mean_zeroone_error_rate(output, target):
    pred = T.round(output)
    return T.neq(pred, target).mean(axis=0)