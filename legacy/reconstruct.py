#!/usr/bin/python2
# -*- coding: utf-8 -*-

import cPickle
import numpy
import theano
import theano.tensor as T
from scipy.spatial.distance import cdist

from reid.preproc import imageproc
from reid.datasets import Datasets
from reid.models.layers import FullConnLayer, ConvPoolLayer
from reid.models.neural_net import NeuralNet, AutoEncoder
from reid.models import active_functions as actfuncs
from reid.models import cost_functions as costfuncs
from reid.optimization import sgd
from reid.utils import data_manager
from reid.utils import cmc
from reid.utils.data_manager import DataLoader


_cached_datasets = '../cache/reconstruct_datasets.pkl'
_cached_model = '../cache/reconstruct_model.pkl'
_cached_output = '../cache/reconstruct_output.pkl'
_cached_distmat = '../cache/reconstruct_distmat.pkl'


def _preproc(image):
    image = imageproc.imresize(image, (80, 40, 3))
    image = numpy.rollaxis(image, 2)
    # image = imageproc.subtract_luminance(image)
    # image = imageproc.scale_per_channel(image, [0, 1])
    return image / 255.0

def _prepare_data(load_from_cache=False, save_to_cache=False):
    if load_from_cache:
        with open(_cached_datasets, 'rb') as f:
            views_data, datasets = cPickle.load(f)
    else:
        image_data = DataLoader('../data/cuhk_small_masked.mat', verbose=True)

        # Prepare the view-first order data representation
        print "Preparing the view-first order data ..."

        n_pedes, n_views = [], []
        for gid in xrange(image_data.get_n_groups()):
            m, v = image_data.get_n_pedes_views(gid)
            n_pedes.append(m)
            n_views.append(v)

        assert min(n_views) == max(n_views), \
            "The number of views in each group should be equal"

        v = n_views[0]

        views_data = [[] for __ in xrange(v)]
        for gid in xrange(image_data.get_n_groups()):
            bias = sum(n_pedes[0:gid])
            group_data = data_manager.view_repr(image_data.get_pedes(gid))

            for vid in xrange(v):
                view_data = group_data[vid]
                view_data = [(pid+bias, image) for pid, image in view_data]
                views_data[vid].extend(view_data)

        # Prepare the datasets
        print "Prepare the datasets ..."

        X, Y = [], []
        for gid in xrange(image_data.get_n_groups()):
            m, v = image_data.get_n_pedes_views(gid)

            for pid in xrange(m):
                n_images = image_data.get_n_images(gid, pid)

                for vi in xrange(v):
                    for vj in xrange(vi+1, v):
                        for i in xrange(n_images[vi]):
                            for j in xrange(n_images[vj]):
                                X.append(_preproc(
                                    image_data.get_image(gid, pid, vi, i)))
                                Y.append(_preproc(
                                    image_data.get_image(gid, pid, vj, j)))

        X = imageproc.images2mat(X).astype(theano.config.floatX)
        Y = imageproc.images2mat(Y).astype(theano.config.floatX)

        datasets = Datasets(X, Y)
        datasets.split(train_ratio=0.8, valid_ratio=0.1)

    if save_to_cache:
        with open(_cached_datasets, 'wb') as f:
            cPickle.dump((views_data, datasets), f,
                protocol=cPickle.HIGHEST_PROTOCOL)

    return (views_data, datasets)

def _train_model(datasets, load_from_cache=False, save_to_cache=False):
    if load_from_cache:
        with open(_cached_model, 'rb') as f:
            model = cPickle.load(f)
    else:
        # Build model
        print "Building model ..."

        # model = AutoEncoder(layer_sizes=[9600, 2400, 2400, 2400, 2400],
        #                     active_funcs=[actfuncs.sigmoid, actfuncs.sigmoid, actfuncs.sigmoid, actfuncs.sigmoid],
        #                     cost_func=costfuncs.mean_square_error,
        #                     error_func=costfuncs.mean_square_error)

        layers = [ConvPoolLayer((128,3,5,5), (2,2), (3,80,40), actfuncs.sigmoid, False),
                  ConvPoolLayer((64,128,5,5), (1,1), None, actfuncs.sigmoid, False),
                  ConvPoolLayer((32,64,5,5), (1,1), None, actfuncs.sigmoid, True),
                  FullConnLayer(9600, 9600, actfuncs.sigmoid),
                  FullConnLayer(9600, 9600, actfuncs.sigmoid)]

        model = NeuralNet(layers, costfuncs.mean_square_error, costfuncs.mean_square_error)

        import copy
        pretrain_datasets = copy.copy(datasets)
        pretrain_datasets.train_y = pretrain_datasets.train_x
        pretrain_datasets.valid_y = pretrain_datasets.valid_x
        pretrain_datasets.test_y = pretrain_datasets.test_x

        sgd.train(model, pretrain_datasets, n_epoch=10, learning_rate=1e-3, learning_rate_decr=1.0)

        sgd.train(model, datasets, n_epoch=100, learning_rate=1e-3, learning_rate_decr=1.0)

    if save_to_cache:
        with open(_cached_model, 'wb') as f:
            cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)

    return model

def _get_distance(model, data, load_from_cache=False, save_to_cache=False):
    if load_from_cache:
        with open(_cached_distmat, 'rb') as f:
            D, G, P = cPickle.load(f)
    else:
        print "Computing distance matrix ..."

        x = T.matrix('x')
        y = model.get_output(x)
        output_func = theano.function(inputs=[x], outputs=y)

        X = [_preproc(image) for pid, image in data[0]]
        Y = [_preproc(image) for pid, image in data[1]]

        X = imageproc.images2mat(X).astype(theano.config.floatX)
        Y = imageproc.images2mat(Y).astype(theano.config.floatX)

        R = [0] * X.shape[0]
        for i in xrange(X.shape[0]):
            R[i] = output_func(X[i:i+1, :]).ravel()
        R = numpy.asarray(R)

        D = cdist(R, Y, 'euclidean')

        G = numpy.asarray([pid for pid, image in data[0]])
        P = numpy.asarray([pid for pid, image in data[1]])

    if save_to_cache:
        with open(_cached_distmat, 'wb') as f:
            cPickle.dump((D, G, P), f, protocol=cPickle.HIGHEST_PROTOCOL)

    with open(_cached_output, 'wb') as f:
        cPickle.dump((X, Y, R), f, protocol=cPickle.HIGHEST_PROTOCOL)

    return (D, G, P)

if __name__ == '__main__':
    views_data, datasets = _prepare_data(load_from_cache=True,
                                         save_to_cache=False)

    model = _train_model(datasets, load_from_cache=False, save_to_cache=True)

    distmat, glabels, plabels = _get_distance(model, views_data,
                                              load_from_cache=False,
                                              save_to_cache=True)

    print cmc.count(distmat, glabels, plabels, 100)

