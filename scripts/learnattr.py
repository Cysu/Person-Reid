#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
import numpy
import attrconf
import bodyconf
from reid.utils.cache_manager import CacheManager


cachem = CacheManager(os.path.join('..', 'cache'), 'attr')


@cachem.save('raw')
def load_data(datasets):
    """Load images, corresponding body parts maps and attributes vectors

    Args:
        datasets: A list of datasets names
    
    Returns:
        A list of tuples. Each is for one pedestrian in the form of
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
def decompose(rawdata, dilation_radius=3):
    """Decompose pedestrain into several body parts groups and 

    This function will generate an image containing only the region of
    particular body parts for each group.

    Args:
        rawdata: A list of pedestrian tuples returned by ``load_data``
        dilation_radius: The radius of dilation to be performed on group region

    Returns:
        A list of tuples. Each is for one pedestrian in the form of
        ``([img_0, img_1, ... , img_m], attribute)`` where ``m`` is the number
        of body parts groups.
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

        # Decompose attributes vector into uni- and multi- value groups
        def decomp_attr(attr):
            attrs = []

            for grp in attrconf.unival:
                subattr = attr[[attrconf.names.index(s) for s in grp]]
                if subattr.sum() != 1: return None
                attrs.append(numpy.where(subattr == 1)[0])

            for grp in attrconf.multival:
                subattr = attr[[attrconf.names.index(s) for s in grp]]
                if subattr.sum() == 0: return None
                attrs.append(subattr)

            return attrs

        data = []
        for img, bpmap, attr in rawdata:
            attr = decomp_attr(attr)
            if attr is None: continue
            body = decomp_body(img, bpmap)
            data.append((body, attr))

    return data


@cachem.save('dataset')
def create_dataset(data):
    """Create dataset for model training, validation and testing

    This function will pre-process the decomposed images and flatten each sample
    into a vector.

    Args:
        data: A list of pedestrian tuples returned by ``decomp_body``

    Returns:
        A Dataset object to be used for model training, validation and
        testing.
    """

    dataset = cachem.load('dataset')

    if dataset is None:
        from reid.utils.dataset import Dataset
        from reid.preproc import imageproc

        def imgprep(img):
            img = imageproc.subtract_luminance(img)
            img = numpy.rollaxis(img, 2)
            return numpy.tanh(img)

        m = len(data)
        X, Y = [0] * m, [0] * m

        for i, (imgs, attr) in enumerate(data):
            X[i] = [imgprep(img) for img in imgs]
            X[i] = numpy.asarray(X[i], dtype=numpy.float32).ravel()
            Y[i] = numpy.concatenate(attr).astype(numpy.float32)

        X = numpy.asarray(X)
        Y = numpy.asarray(Y)

        dataset = Dataset(X, Y)

    return dataset


@cachem.save('model')
def train_model(dataset):
    """Train deep model

    This function will build up a deep neural network and train it using given
    dataset.

    Args:
        dataset: A Dataset object returned by ``create_dataset``

    Returns:
        The trained deep model.
    """

    model = cachem.load('model')

    if model is None:
        import reid.models.active_functions as actfuncs
        from reid.models.layers import ConvPoolLayer, CompLayer, DecompLayer
        from reid.models.neural_net import NeuralNet, MultiwayNeuralNet

        input_decomp = DecompLayer([(3,160,80)] * 4)

        columns = MultiwayNeuralNet([NeuralNet([
            ConvPoolLayer((64,3,5,5), (2,2), (3,160,60), actfuncs.tanh, False),
            ConvPoolLayer((64,3,5,5), (2,2), None, actfuncs.tanh, False),
            ConvPoolLayer((64,3,5,5), (2,2), None, actfuncs.tanh, True)
        ]) for __ in xrange(len(bodyconf.groups))])

        feature_comp = CompLayer()



    return model


if __name__ == '__main__':
    data = load_data(attrconf.datasets)
    data = decompose(data)
    data = create_dataset(data)

    # model = train_model(data)
