#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy
import theano
import theano.tensor as T


class Evaluator(object):
    """Evaluate neural network by computing cost and error"""

    def __init__(self, model, cost_func, error_func, adapter=None,
                 regularize=0, norm=2):
        """Initialize the Evaluator

        Args:
            model: A NeuralNet model processing input to get output
            cost_func: Cost functions
            error_func: Error functions
            adapter: A NeuralNet model processing target to match the cost
                requirement. ``None`` when no processing is needed.
            regularize: Coefficient of regularization terms in cost
            norm: The norm of parameters to be used for regularization

        When ``cost_func`` and ``error_func`` both are functions, the model
        and ``adapter`` should output only one theano tensor. When ``cost_func``
        and ``error_func`` both are list of functions, the model and ``adapter``
        should output a list of theano tensors respectively.
        """

        super(Evaluator, self).__init__()
        
        self._model = model
        self._cost_func = cost_func
        self._error_func = error_func
        self._adapter = adapter
        self._regularize = regularize
        self._norm = norm

    def get_cost_updates(self, x, target, learning_rate, momentum):
        """Get the cost value and tensor update list for model training

        Args:
            x: The input of the model
            target: The target of the model
            learning_rate: A scalar controling the learning rate
            momentum: A scalar controling the momentum

        Returns:
            Tuple (cost, inc_updates, param_updates)
        """

        # Compute the output
        y = self._model.get_output(x)

        # Process the target
        if self._adapter is not None:
            target = self._adapter.get_output(target)

        # Compute the cost value
        if type(self._cost_func) is list:
            cost = sum([self._cost_func[i](output=o, target=t)
                        for i, (o, t) in enumerate(zip(y, target))])
        else:
            cost = self._cost_func(output=y, target=target)

        if self._regularize > 0:
            cost += self._regularize * self._model.get_regularization(2)

        # Compute the gradients
        grads = T.grad(cost, self._model.parameters)

        # Compute the updates
        create_empty = lambda p: theano.shared(
            numpy.zeros(p.get_value(borrow=True).shape, dtype=p.dtype),
            borrow=True
        )

        incs = [create_empty(p) for p in self._model.parameters]

        param_updates = []
        inc_updates = []

        for p, g, inc in zip(self._model.parameters, grads, incs):
            inc_updates.append((inc, momentum*inc - learning_rate*g))
            param_updates.append((p, p + inc))

        return (cost, inc_updates, param_updates)

    def get_error(self, x, target):
        """Get the error value

        Args:
            x: The input of the model
            target: The target of the model

        Returns:
            The scalar error value
        """

        # Compute the output
        y = self._model.get_output(x)

        # Process the target
        if self._adapter is not None:
            target = self._adapter.get_output(target)

        # Compute the error
        if type(self._error_func) is list:
            error = sum([self._error_func[i](output=o, target=t)
                         for i, (o, t) in enumerate(zip(y, target))])
        else:
            error = self._error_func(output=y, target=target)

        return error
