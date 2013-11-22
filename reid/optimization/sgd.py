#!/usr/bin/python2
# -*- coding: utf-8 -*-

import time
import numpy
import theano
import theano.tensor as T


def train(model, datasets,
          batch_size=10, n_epoch=100, learning_rate=1e-4, valid_freq=None):

    # Setup parameters

    n_batches = datasets.get_train_size() / batch_size

    if valid_freq is None: valid_freq = n_batches


    # Setup training, validation and testing functions

    x = T.matrix('x') # input data
    y = T.matrix('y') # corresponding targets
    i = T.lscalar('i') # batch index

    cost, updates = model.get_cost_updates(x, y, learning_rate)
    error = model.get_error(x, y)

    train_func = theano.function(
        inputs=[i], outputs=cost, updates=updates,
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

    begin_time = time.clock()
    
    print "Start training ..."

    for epoch in xrange(n_epoch):
        print "epoch {0}".format(epoch)

        for j in xrange(n_batches):
            cur_iter = (epoch - 1) * n_batches + j

            # train
            batch_cost = train_func(j)
            print "[train] batch {0}/{1}, iter {2}, cost {3}".format(
                j+1, n_batches, cur_iter, batch_cost)

            # validate
            if (cur_iter + 1) % valid_freq == 0:
                valid_error = valid_func()
                print "[valid] error {0}".format(valid_error)

                # test
                if valid_error < best_valid_error:
                    best_valid_error = valid_error

                    test_error = test_func()
                    print "[test] error {0}".format(test_error)

    end_time = time.clock()

    print "Training complete, time {0}".format(end_time - begin_time)
    print "Best validation error {0}, test error {1}".format(
        best_valid_error, test_error)

