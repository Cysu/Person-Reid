#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy
import theano


class Datasets(object):

    def __init__(self, X=None, Y=None, train_set=None, valid_set=None, test_set=None):
        if X is not None and Y is not None:
            self.X = X
            self.Y = Y
        elif train_set is not None and valid_set is not None and test_set is not None:
            self.X = numpy.vstack((train_set[0], valid_set[0], test_set[0]))
            self.Y = numpy.vstack((train_set[1], valid_set[1], test_set[1]))

            self.train_x = self._create_shared(train_set[0])
            self.train_y = self._create_shared(train_set[1])
            self.valid_x = self._create_shared(valid_set[0])
            self.valid_y = self._create_shared(valid_set[1])
            self.test_x = self._create_shared(test_set[0])
            self.test_y = self._create_shared(test_set[1])
        else:
            raise ValueError("Invalid argument combination")

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

    def add_train_data(self, x, y):
        self.train_x = numpy.vstack((self.train_x.get_value(borrow=True), x));
        self.train_y = numpy.vstack((self.train_y.get_value(borrow=True), y));
        numpy.random.shuffle(self.train_x)
        numpy.random.shuffle(self.train_y)
        self.train_x = self._create_shared(self.train_x);
        self.train_y = self._create_shared(self.train_y);

    def get_train_size(self):
        return self.train_x.get_value(borrow=True).shape[0]

    def _create_shared(self, x):
        return theano.shared(numpy.asarray(x,
            dtype=theano.config.floatX), borrow=True)
