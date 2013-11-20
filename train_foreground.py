#!/usr/bin/python2
# -*- coding: utf-8 -*-

from reid.datasets import loader


if __name__ == '__main__':

    X, Y = loader.load_parse('data/parse/cuhk_large_labeled_subsampled.mat',
        'data/parse/cuhk_large_labeled_subsampled_parse.mat')

    