#!/usr/bin/python2
# -*- coding: utf-8 -*-

from reid.utils.data_manager import DataManager
from reid.preproc.imageproc import imglist2mat


def _get_images_list(filename):

    dm = DataManager(verbose=True)
    dm.read(filename)

    images = []

    for t in xrange(dm.n_groups()):
        pedes = dm.get_pedes(t)
        for i in xrange(pedes.shape[0]):
            for j in xrange(pedes.shape[1]):
                v = pedes[i, j]
                for k in xrange(v.shape[1]):
                    images.append(v[0, k])

    return images


def load_parse(origin_filename, parse_filename, verbose=False):

    X = _get_images_list(origin_filename)
    Y = _get_images_list(parse_filename)

    X = imglist2mat(X)
    Y = imglist2mat(Y)

    return (X, Y)