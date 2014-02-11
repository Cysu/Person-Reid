#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
from cache_manager import CacheManager


cachem = CacheManager(os.path.join('..', 'cache'), 'attr')


@cachem.save('rawdata')
def load_data(datasets):
    """Load images, corresponding body parts maps and attributes vectors

    Args:
        datasets: A list of datasets names
    
    Returns a list of tuples. Each is for one pedestrian in the form of
    ``(image, body_parts_map, attribute)``.
    """

    data = cachem.load('rawdata')

    if data is None:
        from scipy.io import loadmat

        data = []

        for name in datasets:
            matfp = os.path.join('..', 'data', 'attributes', name + '_parse.mat')
            matdata = loadmat(matfp)

            m, n = matdata['images'].shape
            for i in xrange(m):
                for j in xrange(n):
                    if matdata['images'][i, j].size == 0: break
                    data.append((matdata['images'][i, j],
                                 matdata['bodyparts'][i, j],
                                 matdata['attributes'][i, 0].ravel()))

    return data


if __name__ == '__main__':
    import attrconf

    rawdata = load_data(attrconf.datasets)
