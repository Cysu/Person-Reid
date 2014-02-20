#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy
import theano
from reid.utils.math_utils import numpy_rng


class Dataset(object):
    """Dataset for model training, validation and testing

    The object contains data stored as theano shared variables. Both feature and
    label of a sample are numpy vectors.
    """

    def __init__(self, X=None, Y=None,
                 train_set=None, valid_set=None, test_set=None):
        """Initialize the Dataset

        Args:
            X, Y: Numpy matrix with each row representing a sample
            train_set, valid_set, test_set: Each is a numpy matrix tuple (X, Y)
                representing features and labels

            If ``X`` and ``Y`` are specified, the Dataset will use them as all
            the samples. Otherwise, if ``train_set``, ``valid_set`` and
            ``test_set`` are specified, the Dataset will stack them together to
            as all the samples.
        """

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
        """Split all the samples into training, validation and testing sets
        randomly

        Args:
            train_ratio: The ratio between training and all the samples
            valid_ratio: The ratio between validation and all the samples
        """

        m = self.X.shape[0]

        m_train = int(m * train_ratio)
        m_valid = int(m * valid_ratio)

        p = numpy_rng.permutation(m)

        self.train_ind = p[0 : m_train]
        self.valid_ind = p[m_train : m_train+m_valid]
        self.test_ind = p[m_train+m_valid : ]

        self.train_x = self._create_shared(self.X[self.train_ind, :])
        self.train_y = self._create_shared(self.Y[self.train_ind, :])
        self.valid_x = self._create_shared(self.X[self.valid_ind, :])
        self.valid_y = self._create_shared(self.Y[self.valid_ind, :])
        self.test_x = self._create_shared(self.X[self.test_ind, :])
        self.test_y = self._create_shared(self.Y[self.test_ind, :])

    @staticmethod
    def _create_shared(x):
        return theano.shared(numpy.asarray(x, dtype=theano.config.floatX),
                             borrow=True)

