#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy
import skimage.transform
from skimage.color import rgb2lab
from sklearn.preprocessing import MinMaxScaler, Binarizer


def imtranslate(image, translation):
    trans = skimage.transform.AffineTransform(translation=translation)
    ret = skimage.transform.warp(image, trans.inverse)

    if image.dtype == numpy.uint8:
        ret = (ret * 255).astype(numpy.uint8)

    return ret

def imresize(image, shape, keep_ratio=False):
    """Resize an image to desired shape

    Args:
        image: A numpy 2d/3d array
        shape: A tuple (h, w) representing the desired height and width
        keep_ratio:
            False: The image will be stretched
            'height': The original height/weight ratio will be reserved. Image
                will be scaled to the desired height. Extra columns will be
                either truncated or filled with zero.
            'width': The original height/weight ratio will be reserved. Image
                will be scaled to the desired width. Extra rows will be either
                truncated or filled with zero.
    """

    if image.ndim == 3:
        shape += (image.shape[2],)
    elif image.ndim != 2:
        raise ValueError("Invalid image dimension")

    if keep_ratio == False:
        ret = skimage.transform.resize(image, shape)
    elif keep_ratio == 'height':
        scale = shape[0] * 1.0 / image.shape[0]
        image = skimage.transform.rescale(image, scale)
        width = image.shape[1]

        if width >= shape[1]:
            l = (width - shape[1]) // 2
            if image.ndim == 3:
                ret = image[:, l:shape[1]+l, :]
            elif image.ndim == 2:
                ret = image[:, l:shape[1]+l]
        else:
            l = (shape[1] - width) // 2
            ret = numpy.zeros(shape)
            if image.ndim == 3:
                ret[:, l:width+l, :] = image
            elif image.ndim == 2:
                ret[:, l:width+l] = image
    elif keep_ratio == 'width':
        scale = shape[1] * 1.0 / image.shape[1]
        image = skimage.transform.rescale(image, scale)
        height = image.shape[0]

        if height >= shape[0]:
            l = (height - shape[0]) // 2
            if image.ndim == 3:
                ret = image[l:shape[0]+l, :, :]
            elif image.ndim == 2:
                ret = image[l:shape[0]+l, :]
        else:
            l = (shape[0] - height) // 2
            ret = numpy.zeros(shape)
            if image.ndim == 3:
                ret[l:height+l, :, :] = image
            elif image.ndim == 2:
                ret[l:height+l, :] = image
    else:
        raise ValueError("Invalid argument ``keep_ratio``")

    if image.dtype == numpy.uint8:
        ret = (ret * 255).astype(numpy.uint8)

    return ret

def subtract_luminance(rgbimg):
    labimg = rgb2lab(rgbimg)

    mean_luminance = numpy.mean(labimg[:,:,0])
    labimg[:,:,0] -= mean_luminance

    return labimg

def scale_per_channel(img, scale_range):
    h, w, c = img.shape
    img = img.reshape(h*w, c)

    scaler = MinMaxScaler(scale_range, copy=False)
    img = scaler.fit_transform(img)

    return img.reshape(h, w, c)

def binarize(img, threshold):
    binarizer = Binarizer(threshold, copy=False)
    return binarizer.fit_transform(img)

def images2mat(images):
    return numpy.asarray(map(lambda x: x.ravel(), images))
