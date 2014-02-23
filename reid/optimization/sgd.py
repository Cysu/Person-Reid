#!/usr/bin/python2
# -*- coding: utf-8 -*-

import time
import numpy
import theano
import theano.tensor as T


def train(evaluator, datasets, learning_rate=1e-4, momentum=0,
          batch_size=10, n_epoch=100,
          improvement=1-1e-3, patience_incr=2.0, learning_rate_decr=0.95,
          never_stop=False):
    """Train model with batched Stochastic Gradient Descent(SGD) algorithm

    Args:
        evaluator: An Evaluator object that provides cost, updates and error
        datasets: A Dataset object that provides training, validation and
            testing data
        learning_rate: The initial learning rate
        momentum: The coefficient of momentum term
        batch_size: The batch size
        n_epoch: The number of epoch
        improvement, patience_incr, learning_rate_decr:
            If ``current_valid_error < best_valid_error * improvement``,
            the patience will be updated to ``current_iter * patience_incr``,
            and the learning_rate will be updated to
            ``current_learning_rate * learning_rate_decr``.
        never_stop: When set to True, the training will not stop until user
            interrupts. Otherwise, the training will stop either when all
            the epoch finishes or the patience is consumed.
    """

    # Setup parameters
    n_train_batches = datasets.train_x.get_value(borrow=True).shape[0] / batch_size + 1
    n_valid_batches = datasets.valid_x.get_value(borrow=True).shape[0] / batch_size + 1
    n_test_batches = datasets.test_x.get_value(borrow=True).shape[0] / batch_size + 1

    # Setup training, validation and testing functions
    x = T.matrix('x') # input data
    y = T.matrix('y') # corresponding targets
    i = T.lscalar('i') # batch index
    alpha = T.scalar('alpha')

    # Compute the cost, updates and error
    cost, updates = evaluator.get_cost_updates(x, y, alpha, momentum)
    error = evaluator.get_error(x, y)

    # Build training, validation and testing functions
    train_func = theano.function(
        inputs=[i, alpha], outputs=cost, updates=updates,
        givens={
            x: datasets.train_x[i*batch_size : (i+1)*batch_size],
            y: datasets.train_y[i*batch_size : (i+1)*batch_size]
        })

    valid_func = theano.function(
        inputs=[i], outputs=error,
        givens={
            x: datasets.valid_x[i*batch_size : (i+1)*batch_size],
            y: datasets.valid_y[i*batch_size : (i+1)*batch_size]
        })

    test_func = theano.function(
        inputs=[i], outputs=error,
        givens={
            x: datasets.test_x[i*batch_size : (i+1)*batch_size],
            y: datasets.test_y[i*batch_size : (i+1)*batch_size]
        })

    # Start training
    best_valid_error = numpy.inf
    test_error = numpy.inf

    valid_freq = n_train_batches
    patience = 20 * n_train_batches

    done_looping = False
    
    print "Start training ..."

    begin_time = time.clock()

    for epoch in xrange(n_epoch):
        print "epoch {0}".format(epoch)

        if done_looping: break

        try:
            for j in xrange(n_train_batches):
                cur_iter = epoch * n_train_batches + j

                # train
                batch_cost = train_func(j, learning_rate)
                print "[train] batch {0}/{1}, iter {2}, cost {3}".format(
                    j+1, n_train_batches, cur_iter, batch_cost)

                # validate
                if (cur_iter + 1) % valid_freq == 0:
                    valid_error = numpy.mean(
                        [valid_func(k) for k in xrange(n_valid_batches)])

                    print "[valid] error {0}".format(valid_error)

                    if type(valid_error) is numpy.ndarray:
                        valid_error = valid_error.mean()

                    # test
                    if valid_error < best_valid_error:
                        if valid_error < best_valid_error * improvement:
                            patience = max(patience, cur_iter * patience_incr)
                            learning_rate = learning_rate * learning_rate_decr

                            print "Update patience {0}, learning_rate {1}".format(
                                patience, learning_rate)

                        best_valid_error = valid_error

                        test_error = numpy.mean(
                            [test_func(k) for k in xrange(n_test_batches)])
                        print "[test] error {0}".format(test_error)

                # early stoping
                if cur_iter > patience and not never_stop:
                    done_looping = True
                    break

        except KeyboardInterrupt:
            print "Keyboard interrupt. Stop training."
            done_looping = True

    end_time = time.clock()

    print "Training complete, time {0}".format(end_time - begin_time)
    print "Best validation error {0}, test error {1}".format(
        best_valid_error, test_error)
