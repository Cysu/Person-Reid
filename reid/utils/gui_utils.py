#!/usr/bin/python2
# -*- coding: utf-8 -*-

from PySide.QtGui import QImage


def ndarray2qimage(array):
    """Convert from numpy.ndarray to QImage

    Args:
        array: An numpy.ndarray of size h*w*c. The number of channels could be
               either one or three, one stands for grayscale image, three stands
               for RGB image. The dtype must be numpy.uint32.

    Returns:
        A QImage object constructed from given array.
    """

    array = array.astype('uint32')

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

    data = data.ravel()
    return QImage(data, w, h, QImage.Format_RGB32).copy()
