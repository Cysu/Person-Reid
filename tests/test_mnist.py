#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
import cPickle
from reid.datasets import Datasets
from reid.optimization import sgd
from reid.models import cost_functions as costfuncs
from reid.models import active_functions as actfuncs
from reid.models.layers import FullConnLayer, ConvPoolLayer
from reid.models.neural_net import NeuralNet


def _load_data():
    with open(os.path.join('..', 'data', 'mnist', 'mnist.pkl'), 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)

    train_set = (train_set[0], train_set[1].reshape(train_set[1].shape[0], 1))
    valid_set = (valid_set[0], valid_set[1].reshape(valid_set[1].shape[0], 1))
    test_set = (test_set[0], test_set[1].reshape(test_set[1].shape[0], 1))

    return Datasets(train_set=train_set, valid_set=valid_set, test_set=test_set)

def test_model(model):
    datasets = _load_data()
    sgd.train(model, datasets,
              costfuncs.mean_negative_loglikelihood,
              costfuncs.mean_number_misclassified,
              batch_size=500, n_epoch=200, learning_rate=0.1, learning_rate_decr=1.0)


if __name__ == '__main__':
    layers = [ConvPoolLayer((20,1,5,5), (2,2), (1,28,28), actfuncs.tanh, False),
              ConvPoolLayer((50,20,5,5), (2,2), None, actfuncs.tanh, True),
              FullConnLayer(800, 500, actfuncs.tanh),
              FullConnLayer(500, 10, actfuncs.softmax)]

    model = NeuralNet(layers)

    test_model(model)
