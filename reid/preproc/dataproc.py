#!/usr/bin/python2
# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA


def whitening(X, n_components=None):
    """Whitening the data

    Args:
        X: A matrix with each row representing a sample
        n_components: None if all the principle components are remained

    Returns:
        The whitened matrix
    """

    if n_components is None:
        n_components = X.shape[0]

    pca = PCA(n_components, copy=False, whiten=True)

    return pca.fit_transform(X)
