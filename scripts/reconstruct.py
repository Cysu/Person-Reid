#!/usr/bin/python2
# -*- coding: utf-8 -*-

import cPickle
import theano

from reid.preproc import imageproc
from reid.datasets import Datasets
from reid.models.neural_net import AutoEncoder
from reid.models import active_functions as actfuncs
from reid.models import cost_functions as costfuncs
from reid.optimization import sgd
from reid.utils.data_manager import DataLoader


_cached_datasets = '../cache/reconstruct_datasets.pkl'
_cached_model = '../cache/reconstruct_model.pkl'


def _prepare_data(load_from_cache=False, save_to_cache=False):
    if load_from_cache:
        with open(_cached_datasets, 'rb') as f:
            datasets = cPickle.load(f)
    else:
        image_data = DataLoader('../data/cuhk_small_mask.mat', verbose=True)

        # Pre-processing
        print "Pre-processing ..."

        def preproc(image):
            image = imageproc.imresize(image, (160, 80, 3))
            image = imageproc.subtract_luminance(image)
            image = imageproc.scale_per_channel(image, [0, 1])
            return image

        X = []
        Y = []

        for gid in xrange(image_data.get_n_groups()):
            m, v = image_data.get_n_pedes_views(gid)

            for pid in xrange(m):
                n_images = image_data.get_n_images(gid, pid)

                for vi in xrange(v):
                    for vj in xrange(vi+1, v):
                        for i in xrange(n_images[vi]):
                            for j in xrange(n_images[vj]):
                                X.append(preproc(image_data.get_image(gid, pid, vi, i)))
                                Y.append(preproc(image_data.get_image(gid, pid, vj, j)))

        X = imageproc.images2mat(X).astype(theano.config.floatX)
        Y = imageproc.images2mat(Y).astype(theano.config.floatX)
        
        # Prepare the datasets
        print "Prepare the datasets ..."

        datasets = Datasets(X, Y)
        datasets.split(train_ratio=0.5, valid_ratio=0.3)

    if save_to_cache:
        with open(_cached_datasets, 'wb') as f:
            cPickle.dump(datasets, f, protocol=cPickle.HIGHEST_PROTOCOL)

    return datasets

def _train_model(datasets, load_from_cache=False, save_to_cache=False):
    if load_from_cache:
        with open(_cached_model, 'rb') as f:
            model, threshold = cPickle.load(f)
    else:
        # Build model
        print "Building model ..."

        model = AutoEncoder(layer_sizes=[38400, 1024, 1024],
                            active_funcs=[actfuncs.sigmoid, actfuncs.sigmoid],
                            cost_func=costfuncs.mean_binary_cross_entropy,
                            error_func=costfuncs.mean_binary_cross_entropy)

        sgd.train(model, datasets, n_epoch=100, learning_rate=1e-4)

    if save_to_cache:
        with open(_cached_model, 'wb') as f:
            cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)

    return model

if __name__ == '__main__':
    datasets = _prepare_data(load_from_cache=True, save_to_cache=False)
    model = _train_model(datasets, load_from_cache=False, save_to_cache=True)
