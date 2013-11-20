#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy
import skimage.color as skicolor


def img2vec(image):

    if image.ndim == 3:
        image = skicolor.rgb2lab(image)
        image = image.swapaxes(0, 2).swapaxes(1, 2)

    return image.flatten()

def imglist2mat(imagelist):

    if imagelist[0].ndim == 3:
        imagelist = numpy.asarray(map(skicolor.rgb2lab, imagelist))
        imagelist = imagelist.swapaxes(1, 3).swapaxes(2, 3)
    else:
        imagelist = numpy.asarray(imagelist)

    return numpy.asarray(map(lambda x: x.flatten(), imagelist))

