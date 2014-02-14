#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
import cPickle


class CacheManager(object):
    """Manage cache saving and loading for a project"""

    def __init__(self, home, tag):
        """Initialize the CacheManager

        Args:
            home: The home where data to be cached
            tag: The tag of the project
        """

        super(CacheManager, self).__init__()

        self.home = home
        self.tag = tag

    def load(self, task):
        """Load from cache if data file exists

        Args:
            task: The task name

        Returns ``None`` if no data file exists. Otherwise, the data will be
        returned.
        """

        try:
            fpath = os.path.join(self.home, self.tag + '_' + task + '.pkl')
            with open(fpath, 'rb') as f:
                data = cPickle.load(f)
        except IOError:
            data = None

        return data

    def save(self, task):
        """Save to cache

        This is a decorator function that decorates on any function. Its return
        value will be saved and then returned again. If the data file already
        exists, then the new data will not be saved.

        Args:
            task: The task name

        Returns the output of decorated function.
        """

        fpath = os.path.join(self.home, self.tag + '_' + task + '.pkl')

        def decorator(func):
            def wrapper(*args, **kwargs):
                data = func(*args, **kwargs)
                if not os.path.isfile(fpath):
                    with open(fpath, 'wb') as f:
                        cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
                return data
            return wrapper
        return decorator
