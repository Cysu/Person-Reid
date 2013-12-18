#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy
from scipy.io import loadmat, savemat

from reid import utils


class DataLoader(object):
    """Data loader (DataLoader)

    The DataLoader class provides input and output functions for data. The data 
    file should follow the unified format that described in README.md.
    """

    def __init__(self, fpath, verbose=False):
        """Initialize the DataLoader

        Args:
            fpath: The path of the data file to be loaded
            verbose: A switch to whether log some information to the screen
        """

        self._verbose = verbose

        self._log("Reading {0} ...".format(fpath))

        # TODO: Handle errors
        self._data = loadmat(fpath)['data']

    def get_n_groups(self):
        """Get the number of cameras-settings groups"""

        return self._data.shape[0]

    def get_n_pedes_views(self, gid):
        """Get the number of pedestrians and views in specific group

        Args:
            gid: The group index

        Returns:
            A tuple (n_pedes, n_views)
        """

        return self._data[gid, 0]['pedes'].shape

    def get_n_images(self, gid, pid):
        """Get the number of images in each view of a pedestrian

        Args:
            gid: The group index
            pid: The pedestrian index

        Returns:
            A list of integers representing the number of images in each view.
        """

        views = self._data[gid, 0]['pedes'][pid, :]
        
        return [v.shape[1] for v in views]

    def get_pedes(self, gid):
        """Get the pedestrian data for one group

        Args:
            gid: The group index

        Returns:
            A m×v numpy matrix `P`, `P(i,j)` is the data array for the `i`-th
            person in `j`-th camera view
        """

        # TODO: Handle errors

        return self._data[gid, 0]['pedes']

    def get_image(self, gid, pid, vid, k):
        """Get a single image for given index

        Args:
            gid: The group index
            pid: The pedestrian index
            vid: The view index
            k: The image index

        Returns:
            The specific image
        """

        return self._data[gid][0]['pedes'][pid, vid][0, k]

    def get_all_images(self):
        """Get all the images as a list

        Returns:
            A list of images
        """

        images = []
        for t in xrange(self._data.shape[0]):
            pedes = self._data[t, 0]['pedes']
            for i in xrange(pedes.shape[0]):
                for j in xrange(pedes.shape[1]):
                    v = pedes[i, j]
                    for k in xrange(v.shape[1]):
                        images.append(v[0, k])

        return images

    def _log(self, msg):
        """Log a message to the screen if verbose mode selected

        Args:
            msg: A message string
        """

        if self._verbose: print msg


class DataSaver(object):

    def __init__(self, verbose=False):
        self._verbose = verbose
        self._data = []

    def save(self, fpath):
        self._log("Reading {0} ...".format(fpath))

        savemat(fpath, {'data': self._py2mat(self._data)})

    def add_group(self, n_pedes, n_views):
        self._data.append([{'pedes': utils.cell([], n_pedes, n_views)}])

        return len(self._data)-1

    def set_images(self, gid, pid, vid, images):
        self._data[gid][0]['pedes'][pid][vid] = images

    def _py2mat(self, py_data):

        # TODO: The code here is so ugly!

        d = len(py_data)
        mat_data = numpy.empty((d, 1), dtype=[('pedes', object)])

        for t in xrange(d):
            py_pedes = py_data[t][0]['pedes']
            m, v = len(py_pedes), len(py_pedes[0])

            mat_data[t, 0]['pedes'] = numpy.empty((m, v), dtype=object)
            mat_pedes = mat_data[t, 0]['pedes']

            for i in xrange(m):
                for j in xrange(v):
                    py_images = py_pedes[i][j]
                    l = len(py_images)
                    mat_pedes[i, j] = numpy.empty((1, l), dtype=object)
                    mat_images = mat_pedes[i, j]

                    for k in xrange(l):
                        mat_images[0, k] = py_images[k]

        return mat_data


    def _log(self, msg):
        """Log a message to the screen if verbose mode selected

        Args:
            msg: A message string
        """

        if self._verbose: print msg
