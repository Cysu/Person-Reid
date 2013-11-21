#!/usr/bin/python2
# -*- coding: utf-8 -*-

from reid.datasets import loader
from reid.preproc import imageproc

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

