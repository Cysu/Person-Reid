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
        A list of tuples with each tuple for one pedestrian in the form of
        ``(img, [img_0, img_1, ... , img_m], attribute)`` where ``img`` is the
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
            data.append((img, body, attr))

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
        from reid.preproc import imageproc, dataproc

        def imgprep(img):
            img = imageproc.imresize(img, (80, 30))
            img = imageproc.subtract_luminance(img)
            img = numpy.rollaxis(img, 2)
            return img

        m = len(data)
        X, Y = [0] * m, [0] * m

        for i, (__, imgs, attr) in enumerate(data):
            X[i] = [imgprep(img) for img in imgs]
            X[i] = numpy.asarray(X[i], dtype=numpy.float32).ravel()
            Y[i] = numpy.concatenate(attr).astype(numpy.float32)

        X = numpy.asarray(X)
        Y = numpy.asarray(Y)

        X = numpy.tanh(dataproc.whitening(X))

        dataset = Dataset(X, Y)
        dataset.split(0.8, 0.1)

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
        import reid.models.cost_functions as costfuncs
        from reid.models.layers import ConvPoolLayer, FullConnLayer
        from reid.models.layers import CompLayer, DecompLayer
        from reid.models.neural_net import NeuralNet, MultiwayNeuralNet
        from reid.models.evaluate import Evaluator
        from reid.optimization import sgd

        # Build up model
        input_decomp = DecompLayer([(3,80,30)] * 4)

        columns = MultiwayNeuralNet([NeuralNet([
            ConvPoolLayer((64,3,3,3), (2,2), (3,80,30), actfuncs.tanh, False),
            ConvPoolLayer((64,64,3,3), (2,2), None, actfuncs.tanh, True)
        ]) for __ in xrange(len(bodyconf.groups))])

        feature_comp = CompLayer(strategy='Maxout')

        classify_1 = FullConnLayer(6912, 99)
        classify_2 = FullConnLayer(99, 99)

        attr_decomp = DecompLayer(
            [(3,), (7,), (2,), (5,), (8,), (10,), (10,), (6,), (11,), (11,), (11,), (15,)],
            [actfuncs.softmax] * 4 + [actfuncs.sigmoid] * 8
        )

        model = NeuralNet([input_decomp, columns, feature_comp, classify_1, classify_2, attr_decomp])

        # Build up adapter
        adapter = DecompLayer(
            [(1,)] * 4 + [(8,), (10,), (10,), (6,), (11,), (11,), (11,), (15,)]
        )

        # Build up evaluator
        cost_functions = [costfuncs.mean_negative_loglikelihood] * 4 + \
                         [costfuncs.mean_binary_cross_entropy] * 8

        error_functions = [costfuncs.mean_number_misclassified] * 4 + \
                          [costfuncs.mean_zeroone_error_rate] * 8

        evaluator = Evaluator(model, cost_functions, error_functions, adapter,
                              regularize=1e-3)

        # Train the model
        sgd.train(evaluator, dataset,
                  learning_rate=1e-3, momentum=0.9,
                  batch_size=300, n_epoch=200,
                  learning_rate_decr=1.0,
                  never_stop=True)

    return model


@cachem.save('result')
def compute_result(model, dataset, data):
    """Compute output value of data samples by using the trained model

    Args:
        model: A NeuralNet model
        dataset: A Dataset object returned by ``create_dataset``
        data:  list of pedestrian tuples returned by ``decomp_body``

    Returns:
        A tuple (train, valid, test) where each is three matrices
        (image_tensor4D, output_matrix, target_matrix)
    """

    result = cachem.load('result')

    if result is None:
        import theano
        import theano.tensor as T
        from reid.models.layers import CompLayer
        from reid.models.neural_net import NeuralNet

        model = NeuralNet([model, CompLayer()])
        x = T.matrix()
        f = theano.function(inputs=[x], outputs=model.get_output(x))

        def compute_output(X):
            outputs = [f(X[i:i+1, :]).ravel() for i in xrange(X.shape[0])]
            return numpy.asarray(outputs)

        images = numpy.asarray([p[0] for p in data])

        train = (images[dataset.train_ind],
                 compute_output(dataset.train_x.get_value(borrow=True)),
                 dataset.train_y.get_value(borrow=True))

        valid = (images[dataset.valid_ind],
                 compute_output(dataset.valid_x.get_value(borrow=True)),
                 dataset.valid_y.get_value(borrow=True))

        test = (images[dataset.test_ind],
                compute_output(dataset.test_x.get_value(borrow=True)),
                dataset.test_y.get_value(borrow=True))

        result = (train, valid, test)

    return result


def show_result(result):
    """Compute the statistics of the result and display them in GUI

    Args:
        result: A list of result tuples returned by ``compute_result``
    """

    train, valid, test = result

    output_seg = [0, 3, 10, 12, 17, 25, 35, 45, 51, 62, 73, 84, 99]
    target_seg = [0, 1, 2, 3, 4, 12, 22, 32, 38, 49, 60, 71, 86]

    def print_stats(title, outputs, targets):
        print "Statistics of {0}".format(title)
        print "=" * 80

        for i, (grptitle, grp) in \
                enumerate(zip(attrconf.unival_titles, attrconf.unival)):
            print "{0}, frequency, accuracy".format(grptitle)

            o = outputs[:, output_seg[i]:output_seg[i+1]]
            t = targets[:, target_seg[i]:target_seg[i+1]]
            p = o.argmax(axis=1).reshape(o.shape[0], 1)

            freqs, accs = [0] * len(grp), [0] * len(grp)
            for j, attrname in enumerate(grp):
                freqs[j] = (t == j).mean()
                accs[j] = ((t == j) & (p == j)).sum() * 1.0 / (t == j).sum()
                if accs[j] is numpy.nan: accs[j] = 0
                print "{0},{1},{2}".format(attrname, freqs[j], accs[j])

            freqs = numpy.asarray(freqs)
            accs = numpy.asarray(accs)

            print "Overall accuracy = {0}".format((freqs * accs).sum())
            print ""

        for i, (grptitle, grp) in \
                enumerate(zip(attrconf.multival_titles, attrconf.multival)):
            print "{0}, frequency, TPR, FPR".format(grptitle)

            o = outputs[:, output_seg[i+4]:output_seg[i+5]]
            t = targets[:, target_seg[i+4]:target_seg[i+5]]
            p = o.round()

            freqs, tprs, fprs = [0] * len(grp), [0] * len(grp), [0] * len(grp)
            for j, attrname in enumerate(grp):
                tj, pj = t[:, j], p[:, j]
                freqs[j] = (tj == 1).mean()
                tprs[j] = ((tj == 1) & (pj == 1)).sum() * 1.0 / (tj == 1).sum()
                fprs[j] = ((tj == 0) & (pj == 1)).sum() * 1.0 / (tj == 0).sum()
                if tprs[j] is numpy.nan: tprs[j] = 0
                if fprs[j] is numpy.nan: fprs[j] = 0
                print "{0},{1},{2},{3}".format(attrname, freqs[j], tprs[j], fprs[j])

            freqs = numpy.asarray(freqs)
            tprs = numpy.asarray(tprs)
            fprs = numpy.asarray(fprs)

            print "Overal TPR = {0}, FPR = {1}".format(
                (freqs * tprs).sum() / freqs.sum(),
                (freqs * fprs).sum() / freqs.sum()
            )
            print ""

    print_stats("Training Set", train[1], train[2])
    print_stats("Validation Set", valid[1], valid[2])
    print_stats("Testing Set", test[1], test[2])


if __name__ == '__main__':
    data = load_data(attrconf.datasets)
    data = decompose(data)
    dataset = create_dataset(data)

    model = train_model(dataset)
    result = compute_result(model, dataset, data)

    show_result(result)
