#!/usr/bin/python2
# -*- coding: utf-8 -*-

import cPickle
import numpy
import theano
import theano.tensor as T

from reid.preproc import imageproc
from reid.utils.data_manager import DataLoader, DataSaver


def _input_preproc(image):
    image = imageproc.imresize(image, (160, 80, 3))
    image = imageproc.subtract_luminance(image)
    image = imageproc.scale_per_channel(image, [0, 1])
    image = numpy.rollaxis(image, 2)
    return image

def _mask_dataset():
    # Load model and compile function
    with open('../cache/foreground_model.pkl', 'rb') as f:
        model, threshold = cPickle.load(f)

    x = T.matrix('x')
    y = model.get_output(x)
    output_func = theano.function(inputs=[x], outputs=(y >= threshold))

    # Load data
    image_data = DataLoader('../data/cuhk_small.mat', verbose=True)

    # Pre-processing
    print "Pre-processing ..."

    images = image_data.get_all_images()
    images = [_input_preproc(image) for image in images]
    images = imageproc.images2mat(images).astype(theano.config.floatX)

    # Compute masks
    print "Computing masks ..."

    masks = output_func(images)

    # Save masks
    print "Saving data ..."

    mask_data = DataSaver()

    cur_index = 0
    for gid in xrange(image_data.get_n_groups()):
        m, v = image_data.get_n_pedes_views(gid)
        mask_data.add_group(m, v)

        for pid in xrange(m):
            n_images = image_data.get_n_images(gid, pid)

            for vid, n in enumerate(n_images):
                view_masks = [0] * n
                for k in xrange(n):
                    mask = masks[cur_index, :]
                    mask = mask.reshape(160, 80, 1)
                    orig_image = image_data.get_image(gid, pid, vid, k)
                    orig_image = imageproc.imresize(orig_image, (160, 80, 3))
                    view_masks[k] = (mask * orig_image).astype(numpy.uint8)
                    cur_index += 1

                mask_data.set_images(gid, pid, vid, view_masks)

    mask_data.save('../data/cuhk_small_masked.mat')

if __name__ == '__main__':
    _mask_dataset()
