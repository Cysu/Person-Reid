#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy


def _cmc_core(D, G, P):
    m, n = D.shape
    order = numpy.argsort(D, axis=0)
    match = (G[order] == P)
    return (match.sum(axis=1) * 1.0 / n).cumsum()

def count(distmat, glabels=None, plabels=None, n_selected_labels=None):
    """Compute the Cumulative Match Characteristic(CMC)

    Args:
        distmat: A ``m×n`` distance matrix. ``m`` and ``n`` are the number of
            gallery and probe samples, respectively. In the case of ``glabels``
            and ``plabels`` both are ``None``, the distance matrix should be
            square and we assume both gallery and probe samples are unique,
            i.e., the i-th gallery samples matches only to the i-th probe
            sample.

        glabels: Vector of length ``m`` that represents the labels of gallery
            samples

        plabels: Vector of length ``n`` that represents the labels of probe
            samples

        n_selected_labels: If specified, we will select only part of all the
            labels to compute the CMC.

    Returns:
        A vector represents the average CMC
    """

    m, n = distmat.shape

    if glabels is None and plabels is None:
        assert m == n
        glabels = numpy.arange(0, m)
        plabels = numpy.arange(0, n)

    unique_glabels = numpy.unique(glabels)

    if n_selected_labels is None:
        if m == unique_glabels.size:
            return _cmc_core(distmat, glabels, plabels)

        n_selected_labels = unique_glabels.size

    n_repeat, ret = 100, 0
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
