#!/usr/bin/python2
# -*- coding: utf-8 -*-

import cPickle
import numpy
import theano
import theano.tensor as T

import reid.optimization.sgd as sgd
from reid.preproc import imageproc
from reid.datasets.datasets import Datasets
from reid.models.neural_net import NeuralNet
from reid.models.layers import FullConnLayer
from reid.models import active_functions as actfuncs
from reid.models import cost_functions as costfuncs
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
        image_data = DataLoader('data/parse/cuhk_large_labeled_subsampled.mat', verbose=True)
        parse_data = DataLoader('data/parse/cuhk_large_labeled_subsampled_parse.mat', verbose=True)

        images = image_data.get_all_images()
        parses = parse_data.get_all_images()

        # Pre-processing
        print "Pre-processing ..."

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
        print "Prepare the datasets ..."
        
        datasets = Datasets(images, parses)
        datasets.split(train_ratio=0.5, valid_ratio=0.3)

    if save_to_cache:
        with open(_cached_datasets, 'wb') as f:
            cPickle.dump(datasets, f, protocol=cPickle.HIGHEST_PROTOCOL)

    return datasets

def _choose_threshold(model, datasets, verbose=False):
    # The trained model output the probability of each pixel to be foreground.
    # We will choose a threshold for it based on ROC curve.

    x = datasets.train_x.get_value(borrow=True)
    target = datasets.train_y.get_value(borrow=True).astype(bool)
    y = theano.function(inputs=[], outputs=model.get_output(x))()

    n_true = (target == True).sum(axis=1)
    n_false = (target == False).sum(axis=1)

    threshold = -1

    def count_roc(t):
        z = numpy.asarray(y >= t)
        fp = ((z == True) & (target == False)).sum(axis=1)
        tp = ((z == True) & (target == True)).sum(axis=1)
        fpr = (fp * 1.0 / n_false).mean()
        tpr = (tp * 1.0 / n_true).mean()
        return (fpr, tpr)

    roc = []
    for t in numpy.arange(1.0, 0.0, -0.01):
        fpr, tpr = count_roc(t)
        roc.append([t, fpr, tpr])
        if threshold == -1 and tpr >= 0.9: threshold = t
    
    if verbose:
        import matplotlib.pyplot as pyplot

        print roc
        roc = numpy.asarray(roc)
        pyplot.plot(roc[:, 1], roc[:, 2])
        pyplot.show()

    print "threshold is chosen as {0}".format(threshold)

    return threshold

def _train_model(datasets, load_from_cache=False, save_to_cache=False):
    if load_from_cache:
        with open(_cached_model, 'rb') as f:
            model, threshold = cPickle.load(f)
    else:
        # Build model
        print "Building model ..."

        layers = [FullConnLayer(input_size=38400,
                                output_size=1024,
                                active_func=actfuncs.sigmoid),

                  FullConnLayer(input_size=1024,
                                output_size=12800,
                                active_func=actfuncs.sigmoid)]

        model = NeuralNet(layers,
                          cost_func=costfuncs.mean_binary_cross_entropy,
                          error_func=costfuncs.mean_binary_cross_entropy)

        sgd.train(model, datasets, n_epoch=100, learning_rate=1e-3)

        threshold = _choose_threshold(model, datasets, verbose=False)

    if save_to_cache:
        with open(_cached_model, 'wb') as f:
            cPickle.dump((model, threshold), f, protocol=cPickle.HIGHEST_PROTOCOL)

    return (model, threshold)


def _generate_result(model, datasets, image_shape, threshold):
    # For convenience, we will save the result in our data format.
    # Regard train, valid, and test sets as three groups.
    # Each pedestrian only has one view, containing output and target images.

    x = T.matrix('x')
    y = model.get_output(x)
    output_func = theano.function(inputs=[x], outputs=y)

    data = DataSaver()
    
    def add(X, Y):
        m = X.shape[0]
        gid = data.add_group(m, 1)

        for pid in xrange(m):
            x, target = X[pid:pid+1, :], Y[pid, :] # Ensure x to be a matrix
            y = (output_func(x) >= threshold).astype(numpy.uint8) * 255
            y = y.reshape(image_shape)
            target = numpy.uint8(target * 255).reshape(image_shape)
            data.set_images(gid, pid, 0, [y, target])

    add(datasets.train_x.get_value(borrow=True),
        datasets.train_y.get_value(borrow=True))
        
    add(datasets.valid_x.get_value(borrow=True),
        datasets.valid_y.get_value(borrow=True))

    add(datasets.test_x.get_value(borrow=True),
        datasets.test_y.get_value(borrow=True))

    data.save(_cached_result)

if __name__ == '__main__':
    datasets = _prepare_data(load_from_cache=True, save_to_cache=False)
    model, threshold = _train_model(datasets, load_from_cache=False, save_to_cache=True)
    _generate_result(model, datasets, (160, 80), threshold)
