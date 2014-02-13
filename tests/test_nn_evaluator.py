#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy
import theano
import theano.tensor as T
from reid.models.cost_functions import mean_square_error as mse
from reid.models.layers import FullConnLayer, DecompLayer, CompLayer
from reid.models.neural_net import MultiwayNeuralNet, NeuralNet
from reid.models.evaluate import Evaluator


# Setup test sample
X = numpy.asarray([[2,1,1,3]], dtype=numpy.float32)
Y = numpy.asarray([[6,9,10,4]], dtype=numpy.float32)
Z = numpy.asarray([[5,8,10,6]], dtype=numpy.float32)
W1 = theano.shared(numpy.asarray([[1,2], [3,4]], dtype=numpy.float32), borrow=True)
W2 = theano.shared(numpy.asarray([[4,3], [2,1]], dtype=numpy.float32), borrow=True)

# Build up model
decomp = DecompLayer([(2,), (2,)])

columns = MultiwayNeuralNet([
    FullConnLayer(2, 2, W=W1),
    FullConnLayer(2, 2, W=W2)
])

comp = CompLayer()

multitask = DecompLayer([(1,), (3,)])

model = NeuralNet([decomp, columns, comp, multitask])

# Build up the target value adapter
adapter = DecompLayer([(1,), (3,)])

# Build up evaluator
evaluator = Evaluator(model, [mse, mse], [mse, mse], adapter)

# Compute the expression by using the model
x = T.matrix('x')
y = T.matrix('y')

output = model.get_output(x)
cost, updates = evaluator.get_cost_updates(x, y, 1.0)
error = evaluator.get_error(x, y)

f = theano.function(inputs=[x, y],
                    outputs=[cost, error])

cost, error = f(X, Y)
print "The target matrix is ", Y
print "The cost is ", cost
print "The error is ", error

cost, error = f(X, Z)
print "The target matrix is ", Z
print "The cost is ", cost
print "The error is ", error