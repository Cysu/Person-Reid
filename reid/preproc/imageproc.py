#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy

from skimage.color import rgb2lab
from sklearn.preprocessing import MinMaxScaler, Binarizer


def subtract_luminance(rgbimg):

    labimg = rgb2lab(rgbimg)

    mean_luminance = numpy.mean(labimg[:,:,0])
    labimg[:,:,0] -= mean_luminance

    return labimg

def scale_per_channel(img, scale_range):

    scaler = MinMaxScaler(scale_range, copy=False)
    return scaler.fit_transform(img)

def binarize(img, threshold):

    binarizer = Binarizer(threshold, copy=False)
    return binarizer.fit_transform(img)

def images2mat(images):
    
    return numpy.asarray(map(lambda x: x.flatten(), images))
