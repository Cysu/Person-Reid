#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy
import theano
import theano.tensor as T
from reid.models.layers import FullConnLayer, DecompLayer, CompLayer
from reid.models.neural_net import MultiwayNeuralNet, NeuralNet


# Setup test sample
X = numpy.asarray([2,1,1,3], dtype=numpy.float32).reshape(1, 4)
Y = numpy.asarray([5,8,10,6], dtype=numpy.float32).reshape(1, 4)
W1 = theano.shared(numpy.asarray([[1,2], [3,4]], dtype=numpy.float32), borrow=True)
W2 = theano.shared(numpy.asarray([[4,3], [2,1]], dtype=numpy.float32), borrow=True)

# Build up model
decomp = DecompLayer([(2,), (2,)])

columns = MultiwayNeuralNet([
    FullConnLayer(2, 2, W=W1),
    FullConnLayer(2, 2, W=W2)
])

comp = CompLayer()

model = NeuralNet([decomp, columns, comp])

# Compute the expression by using the model
inp = T.matrix('input')
f = theano.function(inputs=[inp], outputs=model.get_output(inp))

print "The target matrix is ", Y
print "The output matrix is ", f(X)
