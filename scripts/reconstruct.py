#!/usr/bin/python2
# -*- coding: utf-8 -*-

import cPickle
import numpy
import theano
import theano.tensor as T

from reid.preproc import imageproc
from reid.datasets import Datasets
from reid.models.neural_net import AutoEncoder
from reid.models import active_functions as actfuncs
from reid.models import cost_functions as costfuncs
from reid.optimization import sgd
from reid.utils import data_manager
from reid.utils.data_manager import DataLoader


_cached_datasets = '../cache/reconstruct_datasets.pkl'
_cached_model = '../cache/reconstruct_model.pkl'
_cached_distmat = '../cache/reconstruct_distmat.pkl'


def _preproc(image):
    image = imageproc.imresize(image, (160, 80, 3))
    image = imageproc.subtract_luminance(image)
    image = imageproc.scale_per_channel(image, [0, 1])
    return image

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
        datasets.split(train_ratio=0.5, valid_ratio=0.3)

    if save_to_cache:
        with open(_cached_datasets, 'wb') as f:
            cPickle.dump((views_data, datasets), f, 
                protocol=cPickle.HIGHEST_PROTOCOL)

    return (views_data, datasets)

def _train_model(datasets, load_from_cache=False, save_to_cache=False):
    if load_from_cache:
        with open(_cached_model, 'rb') as f:
            model, threshold = cPickle.load(f)
    else:
        # Build model
        print "Building model ..."

        model = AutoEncoder(layer_sizes=[38400, 1024, 1024],
                            active_funcs=[actfuncs.sigmoid, actfuncs.sigmoid],
                            cost_func=costfuncs.mean_square_error,
                            error_func=costfuncs.mean_square_error)

        sgd.train(model, datasets, n_epoch=100, learning_rate=1e-4)

    if save_to_cache:
        with open(_cached_model, 'wb') as f:
            cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)

    return model

def _get_distance(model, data, load_from_cache=False, save_to_cache=False):
    if load_from_cache:
        with open(_cached_distmat, 'rb') as f:
            D = cPickle.load(f)
    else:
        x = T.matrix('x')
        y = model.get_output(x)
        output_func = theano.function(inputs=[x], outputs=y)

        X = [_preproc(image) for pid, image in data[0]]
        Y = [_preproc(image) for pid, image in data[1]]

        X = imageproc.images2mat(X).astype(theano.config.floatX)
        Y = imageproc.images2mat(Y).astype(theano.config.floatX)

        R = output_func(X)

        nr, ny = R.shape[0], Y.shape[0]
        D = numpy.zeros((nr, ny), dtype=theano.config.floatX)

        for i in xrange(nr):
            for j in xrange(ny):
                D[i, j] = ((R[i]-D[j])**2).sum()

    if save_to_cache:
        with open(_cached_distmat, 'wb') as f:
            cPickle.dump(D, f, protocol=cPickle.HIGHEST_PROTOCOL)

    return D

if __name__ == '__main__':
    views_data, datasets = _prepare_data(load_from_cache=True,
                                         save_to_cache=False)

    model = _train_model(datasets, load_from_cache=False, save_to_cache=True)

    distmat = _get_distance(model, views_data,
                            load_from_cache=False,
                            save_to_cache=True)
