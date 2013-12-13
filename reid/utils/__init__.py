#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy
from PySide.QtGui import QImage


numpy_rng = numpy.random.RandomState(999987)

def cell(default, *args):
    result = [default for __ in xrange(args[0])]

    if len(args) > 1:
        for i in xrange(args[0]):
            result[i] = cell(default, *(args[1:]))

    return result

def ndarray2qimage(array):
    """Convert from numpy.ndarray to QImage

    Args:
        array: An numpy.ndarray of size h*w*c. The number of channels could be 
               either one or three, one stands for grayscale image, three stands
               for RGB image. The dtype must be numpy.uint32.

    Returns:
        A QImage object constructed from given array.
    """

    array = array.astype(numpy.uint32)

    if array.ndim == 2:
        h, w = array.shape
        data = (255 << 24 | array << 16 | array << 8 | array)
    elif array.ndim == 3:
        h, w, c = array.shape
        if c == 1:
            data = (255 << 24 | array[:,:,0] << 16 | array[:,:,0] << 8 | array[:,:,0])
        elif c == 3:
            data = (255 << 24 | array[:,:,0] << 16 | array[:,:,1] << 8 | array[:,:,2])
        else:
            raise ValueError("ndarray2qimage cannot recognize the image type")
    else:
        raise ValueError("ndarray2qimage cannot recognize the image type")

    return QImage(data.ravel(), w, h, QImage.Format_RGB32).copy()