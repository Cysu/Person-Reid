#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy
import theano


class Datasets(object):

    def __init__(self, X, Y):

        assert X.shape[0] == Y.shape[0]

        self.X = X
        self.Y = Y

    def split(self, train_ratio, valid_ratio):

        m = self.X.shape[0]

        m_train = int(m * train_ratio)
        m_valid = int(m * valid_ratio)

        p = numpy.random.permutation(m)

        train_ind = p[0 : m_train]
        valid_ind = p[m_train : (m_train+m_valid)]
        test_ind = p[(m_train+m_valid) : ]

        self.train_x = self._create_shared(self.X[train_ind, :])
        self.train_y = self._create_shared(self.Y[train_ind, :])
        self.valid_x = self._create_shared(self.X[valid_ind, :])
        self.valid_y = self._create_shared(self.Y[valid_ind, :])
        self.test_x = self._create_shared(self.X[test_ind, :])
        self.test_y = self._create_shared(self.Y[test_ind, :])

    def get_train_size(self):

        return self.train_x.get_value(borrow=True).shape[0]

    def _create_shared(x):
        
        return theano.shared(numpy.asarray(x,
            dtype=theano.config.floatX), borrow=True)
