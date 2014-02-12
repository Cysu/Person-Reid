#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
import numpy
from cache_manager import CacheManager


cachem = CacheManager(os.path.join('..', 'cache'), 'attr')


@cachem.save('raw')
def load_data(datasets):
    """Load images, corresponding body parts maps and attributes vectors

    Args:
        datasets: A list of datasets names
    
    Returns a list of tuples. Each is for one pedestrian in the form of
    ``(image, body_parts_map, attribute)``.
    """

    data = cachem.load('raw')

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


@cachem.save('decomp')
def decomp_body(rawdata, groups, dilation_radius=3):
    """Decompose pedestrain into several body parts groups

    This function will generate an image containing only the region of
    particular body parts for each group.

    Args:
        rawdata: A list of pedestrian tuples returned by ``load_data``
        groups: A list of groups' pixel values
        dilation_radius: The radius of dilation to be performed on group region

    Returns a list of tuples. Each is for one pedestrian in the form of
    ``([img_0, img_1, ... , img_m], attribute)`` where ``m`` is the number
    of body parts groups.
    """

    data = cachem.load('decomp')

    if data is None:
        import skimage.morphology as morph
        selem = morph.square(2*dilation_radius)

        data = [0] * len(rawdata)

        for i, (img, bpmap, attr) in enumerate(rawdata):
            imgs = [0] * len(groups)

            for j, grp in enumerate(groups):
                # mask = dilate(region_0 | region_1 | ... | region_k)
                regions = [(bpmap == pixval) for pixval in grp]
                mask = reduce(lambda x, y: x|y, regions)
                mask = morph.binary_dilation(mask, selem)

                imgs[j] = img * numpy.expand_dims(mask, axis=2)

            data[i] = (imgs, attr)

    return data


if __name__ == '__main__':
    import attrconf
    import bodyconf

    data = load_data(attrconf.datasets)
    data = decomp_body(data, bodyconf.groups)
