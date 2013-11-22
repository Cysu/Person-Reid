#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy
import theano.tensor as T

from reid.preproc import imageproc
from reid.datasets import loader
from reid.datasets.datasets import Datasets
from reid.models.mlp import MultiLayerPerceptron as Mlp
from reid.models.layer import Layer

import reid.costs as costs
import reid.optimization.sgd as sgd


if __name__ == '__main__':

    # Setup data files

    image_filename = 'data/parse/cuhk_large_labeled_subsampled.mat'
    parse_filename = 'data/parse/cuhk_large_labeled_subsampled_parse.mat'

    images = loader.get_images_list(image_filename)
    parses = loader.get_images_list(parse_filename)

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

    # Build model

    numpy_rng = numpy.random.RandomState(999987)
    layers = [Layer(numpy_rng, 38400, 1024, T.nnet.sigmoid),
              Layer(numpy_rng, 1024, 12800, T.nnet.sigmoid)]

    model = Mlp(layers,
                cost_func=costs.MeanBinaryCrossEntropy,
                error_func=costs.MeanBinaryCrossEntropy)

    sgd.train(model, datasets)

