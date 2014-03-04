#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
import numpy
import bodyconf
from reid.utils.cache_manager import CacheManager


cachem = CacheManager(os.path.join('..', 'cache'), 'reidbyattr')


@cachem.save('raw')
def load_data(datasets):
    """Load images and corresponding body parts maps

    Divide data into gallery and probes due to the view 0 or 1.

    Args:
        datasets: A list of datasets names

    Returns:
        ``(gallery, probe)`` where each is a list of pedestrians
        ``[image, body_parts_map, pid]``
    """

    data = cachem.load('raw')

    if data is None:
        from scipy.io import loadmat

        for name in datasets:
            matfp = os.path.join('..', 'data', 'reid', name + '_parse.mat')
            matdata = loadmat(matfp)

            m, n = matdata['images'].shape
            assert n == 2, "The number of views should be two"

            gallery, probe = [], []

            cur_pid = 0
            for i in xrange(m):
                for k in xrange(matdata['images'][i, 0].shape[1]):
                    gallery.append([matdata['images'][i, 0][0, k],
                                    matdata['bodyparts'][i, 0][0, k],
                                    cur_pid])
                    cur_pid += 1

            cur_pid = 0
            for i in xrange(m):
                for k in xrange(matdata['images'][i, 1].shape[1]):
                    probe.append([matdata['images'][i, 1][0, k],
                                  matdata['bodyparts'][i, 1][0, k],
                                  cur_pid])
                    cur_pid += 1

        data = (gallery, probe)

    return data


@cachem.save('decomp')
def decompose(rawdata, dilation_radius=3):
    """Decompose pedestrain into several body parts groups

    This function will generate an image containing only the region of
    particular body parts for each group.

    Args:
        rawdata: Pedestrian data returned by ``load_data``
        dilation_radius: The radius of dilation to be performed on group region

    Returns:
        ``(gallery, probe)`` where each is a list of pedestrians
        ``[img, [img_0, img_1, ... , img_m], pid]`` where ``img`` is the
        original image and ``m`` is the number of body parts groups.
    """

    data = cachem.load('decomp')

    if data is None:
        import skimage.morphology as morph
        selem = morph.square(2*dilation_radius)

        # Decompose pedestrian image into body parts
        def decomp_body(img, bpmap):
            imgs = [0] * len(bodyconf.groups)

            for i, grp in enumerate(bodyconf.groups):
                # mask = dilate(region_0 | region_1 | ... | region_k)
                regions = [(bpmap == pixval) for pixval in grp]
                mask = reduce(lambda x, y: x|y, regions)
                mask = morph.binary_dilation(mask, selem)
                imgs[i] = img * numpy.expand_dims(mask, axis=2)

            return imgs

        gallery, probe = rawdata

        for i in xrange(len(gallery)):
            gallery[i][1] = decomp_body(gallery[i][0], gallery[i][1])
        for i in xrange(len(probe)):
            probe[i][1] = decomp_body(probe[i][0], probe[i][1])

        data = (gallery, probe)

    return data


@cachem.save('prep')
def preprocess(data):
    """Preprocess the data

    Args:
        data: Data returned by ``decompose``

    Returns:
        The preprocess dataset
    """

    dataset = cachem.load('prep')

    if dataset is None:
        from reid.preproc import imageproc

        def imgprep(img):
            img = imageproc.imresize(img, (80, 30))
            img = imageproc.subtract_luminance(img)
            img = numpy.rollaxis(img, 2)
            return img

        gallery, probe = data

        gallery_X = [0] * len(gallery)
        gallery_Y = [pid for __, __, pid in gallery]
        for i, (__, imgs, __) in enumerate(gallery):
            gallery_X[i] = [imgprep(img) for img in imgs]
            gallery_X[i] = numpy.asarray(gallery_X[i], dtype=numpy.float32).ravel()

        probe_X = [0] * len(probe)
        probe_Y = [pid for __, __, pid in probe]
        for i, (__, imgs, __) in enumerate(probe):
            probe_X[i] = [imgprep(img) for img in imgs]
            probe_X[i] = numpy.asarray(probe_X[i], dtype=numpy.float32).ravel()

        gallery_X = numpy.asarray(gallery_X)
        gallery_Y = numpy.asarray(gallery_Y)
        probe_X = numpy.asarray(probe_X)
        probe_Y = numpy.asarray(probe_Y)

        gallery_X = numpy.tanh(gallery_X - gallery_X.mean(axis=0))
        probe_X = numpy.tanh(probe_X - probe_X.mean(axis=0))

        dataset = (gallery_X, gallery_Y, probe_X, probe_Y)

    return dataset


@cachem.save('attr')
def compute_attr(model, dataset, batch_size=100):
    """Compute the attributes matrix

    Args:
        model: The deep neural net model
        dataset: The dataset returned by ``preprocess``

    Returns:
        The dataset ``(gA, gY, pA, pY)``
    """

    data = cachem.load('attr')

    if data is None:
        import theano
        import theano.tensor as T
        from reid.models.layers import CompLayer
        from reid.models.neural_net import NeuralNet

        model = NeuralNet([model, CompLayer()])

        def compute(X):
            x = T.matrix('x')
            i = T.lscalar('i')

            func = theano.function(
                inputs=[i], outputs=model.get_output(x),
                givens={
                    x: X[i*batch_size : (i+1)*batch_size]
                })

            n_batches = X.get_value(borrow=True).shape[0] / batch_size + 1
            return numpy.vstack([func(j) for j in xrange(n_batches)])

        gX, gY, pX, pY = dataset
        gA = compute(theano.shared(gX, borrow=True))
        pA = compute(theano.shared(pX, borrow=True))

        data = (gA, gY, pA, pY)

    return data


@cachem.save('dist')
def compute_distance(dataset):
    """Compute pairwise distance

    Args:
        dataset: The dataset returned by ``compute_attr``

    Returns:
        The pairwise distance matrix D
    """

    from sklearn.metrics.pairwise import pairwise_distances as pwdist

    gA, gY, pA, pY = dataset

    return pwdist(gA, pA, 'euclidean')


if __name__ == '__main__':
    data = load_data(['CUHKL'])
    data = decompose(data)
    dataset = preprocess(data)

    import cPickle
    with open('../cache/attr_model.pkl', 'rb') as f:
        model = cPickle.load(f)

    dataset = compute_attr(model, dataset)

    D = compute_distance(dataset)

    from reid.utils import cmc
    cmcret = cmc.count(D, dataset[1], dataset[3], 100)

    print cmcret