#!/usr/bin/python2
# -*- coding: utf-8 -*-

import time
import numpy
import theano
import theano.tensor as T


def train(model, datasets, cost_func, error_func,
          batch_size=10, n_epoch=100, learning_rate=1e-4, momentum=0,
          improvement=1-1e-3, patience_incr=2.0, learning_rate_decr=0.95):
    # Setup parameters
    n_batches = datasets.get_train_size() / batch_size

    # Setup training, validation and testing functions
    x = T.matrix('x') # input data
    y = T.matrix('y') # corresponding targets
    i = T.lscalar('i') # batch index
    alpha = T.scalar('alpha')

    from reid.models.neural_net import get_cost_updates, get_error

    cost, updates = get_cost_updates(model, cost_func, x, y, alpha, momentum)
    error = get_error(model, error_func, x, y)

    train_func = theano.function(
        inputs=[i, alpha], outputs=cost, updates=updates,
        givens={
            x: datasets.train_x[i*batch_size : (i+1)*batch_size],
            y: datasets.train_y[i*batch_size : (i+1)*batch_size]
        })

    valid_func = theano.function(
        inputs=[], outputs=error,
        givens={
            x: datasets.valid_x,
            y: datasets.valid_y
        })

    test_func = theano.function(
        inputs=[], outputs=error,
        givens={
            x: datasets.test_x,
            y: datasets.test_y
        })

    # Start training
    best_valid_error = numpy.inf
    test_error = numpy.inf

    valid_freq = n_batches
    patience = 20 * n_batches

    done_looping = False
    
    print "Start training ..."

    begin_time = time.clock()

    for epoch in xrange(n_epoch):
        print "epoch {0}".format(epoch)

        if done_looping: break

        try:
            for j in xrange(n_batches):
                cur_iter = epoch * n_batches + j

                # train
                batch_cost = train_func(j, learning_rate)
                print "[train] batch {0}/{1}, iter {2}, cost {3}".format(
                    j+1, n_batches, cur_iter, batch_cost)

                # validate
                if (cur_iter + 1) % valid_freq == 0:
                    valid_error = valid_func()
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

                        test_error = test_func()
                        print "[test] error {0}".format(test_error)

                # early stoping
                if cur_iter > patience:
                    done_looping = True
                    break

        except KeyboardInterrupt:
            print "Keyboard interrupt. Stop training."
            done_looping = True

    end_time = time.clock()

    print "Training complete, time {0}".format(end_time - begin_time)
    print "Best validation error {0}, test error {1}".format(
        best_valid_error, test_error)
