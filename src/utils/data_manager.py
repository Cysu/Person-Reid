#! /usr/bin/python2
# -*- coding: utf-8 -*-

from scipy.io import loadmat


class DataManager(object):
    """Data I/O manager (DataManager)

    The DataManager class provides input and output functions for data. The data 
    file should follow the unified format that described in README.md.
    """

    def __init__(self, verbose=False):
        """Initialize the DataManager

        Args:
            verbose: A switch to whether log some information to the screen
        """

        self._verbose = verbose

    def read(self, fpath):
        """Read data from file

        Args:
            fpath: The path to the data file
        """

        self._log("Reading {0} ...".format(fpath))

        # TODO: Handle errors
        self._data = loadmat(fpath)['data']

    def n_groups(self):
        """Get the number of cameras-settings groups"""

        return self._data.shape[0]

    def get_pedes(self, index):
        """Get the pedestrian data for one group

        Args:
            index: The group index

        Returns:
            A m×v numpy matrix `P`, `P(i,j)` is the data array for the `i`-th
            person in `j`-th camera view
        """

        # TODO: Handle errors

        return self._data[index, 0]['pedes']

    def _log(self, msg):
        """Log a message to the screen if verbose mode selected

        Args:
            msg: A message string
        """

        if self._verbose: print msg
