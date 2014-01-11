#!/usr/bin/python2
# -*- coding: utf-8 -*-

from reid.preproc import imageproc

def aug_translation(images, targets, offset_x=1, offset_y=1, padding=0):
    """Augment data by image translation

    Args:
        images: A list of images in numpy ndarray format
        targets: A list of corresponding target outputs
        offset_x: An integer representing the translation offset in x axis
        offset_y: An integer representing the translation offset in y axis
        padding: Either a float represting the padding value, or one of the
            strings 'circular', 'replicate' and 'symmetirc'. Only padding with 
            value 0 supported in current version.
    """

    if len(images) != len(targets):
        raise ValueError("images and targets should have same length")

    if type(padding) is str and \
            padding not in ('cirular', 'replicate', 'symmetirc'):
        raise ValueError("Invalid padding option")

    aug_images, aug_targets = [], []

    for image, target in zip(images, targets):
        for dx in xrange(-offset_x, offset_x+1):
            for dy in xrange(-offset_y, offset_y+1):
                aug_images.append(imageproc.imtranslate(image, (dx, dy)))
                aug_targets.append(target)

    return (aug_images, aug_targets)
