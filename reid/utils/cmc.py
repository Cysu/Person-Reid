#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy


def _cmc_core(D, G, P):
    m, n = D.shape
    order = numpy.argsort(D, axis=0)
    match = (G[order] == P)
    return (match.sum(axis=1) * 1.0 / n).cumsum()

def count(distmat, glabels=None, plabels=None, n_selected_labels=None, n_repeat=100):
    """Compute the Cumulative Match Characteristic(CMC)

    Args:
        distmat: A ``m×n`` distance matrix. ``m`` and ``n`` are the number of
            gallery and probe samples, respectively. In the case of ``glabels``
            and ``plabels`` both are ``None``, we assume both gallery and probe
            samples to be unique, i.e., the i-th gallery samples matches only to
            the i-th probe sample.

        glabels: Vector of length ``m`` that represents the labels of gallery
            samples

        plabels: Vector of length ``n`` that represents the labels of probe
            samples

        n_selected_labels: If specified, we will select only part of all the
            labels to compute the CMC.

        n_repeat: The number of random sampling times

    Returns:
        A vector represents the average CMC
    """

    m, n = distmat.shape

    if glabels is None and plabels is None:
        glabels = numpy.arange(0, m)
        plabels = numpy.arange(0, n)

    if type(glabels) is list:
        glabels = numpy.asarray(glabels)
    if type(plabels) is list:
        plabels = numpy.asarray(plabels)

    unique_glabels = numpy.unique(glabels)

    if n_selected_labels is None:
        n_selected_labels = unique_glabels.size

    ret = 0
    for r in xrange(n_repeat):
        # Randomly select gallery labels
        ind = numpy.random.choice(unique_glabels.size,
                                  n_selected_labels,
                                  replace=False)
        ind.sort()
        g = unique_glabels[ind]

        # Select corresponding probe samples
        ind = []
        for i, label in enumerate(plabels):
            if label in g: ind.append(i)
        ind = numpy.asarray(ind)

        p = plabels[ind]

        # Randomly select one sample per selected label
        subdist = numpy.zeros((n_selected_labels, p.size))
        for i, glabel in enumerate(g):
            samples = numpy.where(glabels == glabel)[0]
            j = numpy.random.choice(samples)
            subdist[i, :] = distmat[j, ind]

        # Compute CMC
        ret += _cmc_core(subdist, g, p)

    return ret / n_repeat

def count_lazy(distfunc, glabels=None, plabels=None, n_selected_labels=None, n_repeat=100):
    """Compute the Cumulative Match Characteristic(CMC) in a lazy manner

    This function will only compute the distance when needed.

    Args:
        distfunc: A distance computing function. Denote the number of gallery
            and probe samples by ``m`` and ``n``, respectively.
            ``distfunc(i, j)`` should output distance between gallery sample
            ``i`` and probe sample ``j``. In the case of ``glabels``
            and ``plabels`` both are integers, ``m`` should be equal to ``n``
            and we assume both gallery and probe samples to be unique,
            i.e., the i-th gallery samples matches only to the i-th probe
            sample.

        glabels: Vector of length ``m`` that represents the labels of gallery
            samples. Or an integer ``m``.

        plabels: Vector of length ``n`` that represents the labels of probe
            samples. Or an integer ``n``.

        n_selected_labels: If specified, we will select only part of all the
            labels to compute the CMC.

        n_repeat: The number of random sampling times

    Returns:
        A vector represents the average CMC
    """

    if type(glabels) is int:
        m = glabels
        glabels = numpy.arange(0, m)
    elif type(glabels) is list:
        glabels = numpy.asarray(glabels)
        m = glabels.size
    else:
        m = glabels.size

    if type(plabels) is int:
        n = plabels
        plabels = numpy.arange(0, n)
    elif type(plabels) is list:
        plabels = numpy.asarray(plabels)
        n = plabels.size
    else:
        n = plabels.size

    unique_glabels = numpy.unique(glabels)

    if n_selected_labels is None:
        n_selected_labels = unique_glabels.size

    ret = 0
    for r in xrange(n_repeat):
        # Randomly select gallery labels
        ind = numpy.random.choice(unique_glabels.size,
                                  n_selected_labels,
                                  replace=False)
        ind.sort()
        g = unique_glabels[ind]

        # Select corresponding probe samples
        ind = []
        for i, label in enumerate(plabels):
            if label in g: ind.append(i)
        ind = numpy.asarray(ind)

        p = plabels[ind]

        # Randomly select one sample per selected label
        subdist = numpy.zeros((n_selected_labels, p.size))
        for i, glabel in enumerate(g):
            samples = numpy.where(glabels == glabel)[0]
            j = numpy.random.choice(samples)
            for k in xrange(p.size):
                subdist[i, k] = distfunc(j, ind[k])

        # Compute CMC
        ret += _cmc_core(subdist, g, p)

    return ret / n_repeat
