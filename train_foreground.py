#!/usr/bin/python2
# -*- coding: utf-8 -*-

import cPickle
import numpy
import theano
import theano.tensor as T

import reid.costs as costs
import reid.optimization.sgd as sgd
from reid.preproc import imageproc
from reid.datasets.datasets import Datasets
from reid.models.mlp import MultiLayerPerceptron as Mlp
from reid.models.layer import Layer
from reid.models import active_functions as actfuncs
from reid.utils.data_manager import DataLoader, DataSaver


_cached_datasets = 'cache/foreground_datasets.pkl'
_cached_model = 'cache/foreground_model.pkl'
_cached_result = 'cache/foreground_result.mat'


def _prepare_data(load_from_cache=False, save_to_cache=False):

    if load_from_cache:
        with open(_cached_datasets, 'rb') as f:
            datasets = cPickle.load(f)
    else:
        # Setup data files

        image_data = DataLoader('data/parse/cuhk_large_labeled_subsampled.mat')
        parse_data = DataLoader('data/parse/cuhk_large_labeled_subsampled_parse.mat')

        images = image_data.get_all_images()
        parses = parse_data.get_all_images()

        # Pre-processing

        for i, image in enumerate(images):
            image = imageproc.subtract_luminance(image)
            image = imageproc.scale_per_channel(image, [0, 1])
            images[i] = image

        images = imageproc.images2mat(images)

        for i, parse in enumerate(parses):
            parse = imageproc.binarize(parse, 0)
            parses[i] = parse

        parses = imageproc.images2mat(parses)

        # Prepare the datasets
        
        datasets = Datasets(images, parses)
        datasets.split(train_ratio=0.5, valid_ratio=0.3)

    if save_to_cache:
        with open(_cached_datasets, 'wb') as f:
            cPickle.dump(datasets, f, protocol=cPickle.HIGHEST_PROTOCOL)

    return datasets

def _train_model(datasets, load_from_cache=False, save_to_cache=False):

    if load_from_cache:
        with open(_cached_model, 'rb') as f:
            model = cPickle.load(f)
    else:
        # Build model

        numpy_rng = numpy.random.RandomState(999987)
        layers = [Layer(numpy_rng, 38400, 1024, actfuncs.sigmoid),
                  Layer(numpy_rng, 1024, 12800, actfuncs.sigmoid)]

        model = Mlp(layers,
                    cost_func=costs.MeanBinaryCrossEntropy,
                    error_func=costs.MeanBinaryCrossEntropy)

        sgd.train(model, datasets)

    if save_to_cache:
        with open(_cached_model, 'wb') as f:
            cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)

    return model

def _generate_result(model, datasets):
    # For convenience, we will save the result in our data format.
    # Regard train, valid, and test sets as three groups.
    # Each pedestrian only has one view, containing output and target images.

    x = T.vector('x')
    y = model.get_outputs(x)[-1]
    output_func = theano.function(inputs=[x], outputs=y)

    data = DataSaver()
    
    def add(X, Y):
        m = X.shape[0]
        gid = data.add_group(m, 1)

        for pid in xrange(m):
            x, target = X[pid, :], Y[pid, :]
            y = output_func(x)

            y = numpy.uint8(y * 255).reshape(160, 80)
            target = numpy.uint8(target * 255).reshape(160, 80)

            data.set_images(gid, pid, 0, [y, target])

    add(datasets.train_x.get_value(borrow=True),
        datasets.train_y.get_value(borrow=True))
        
    add(datasets.valid_x.get_value(borrow=True),
        datasets.valid_y.get_value(borrow=True))

    add(datasets.test_x.get_value(borrow=True),
        datasets.test_y.get_value(borrow=True))

    data.save(_cached_result)

if __name__ == '__main__':

    datasets = _prepare_data(load_from_cache=False, save_to_cache=True)

    model = _train_model(datasets, load_from_cache=False, save_to_cache=True)

    _generate_result(model, datasets)
