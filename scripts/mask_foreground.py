#!/usr/bin/python2
# -*- coding: utf-8 -*-

import cPickle
import theano
import theano.tensor as T

from reid.preproc import imageproc
from reid.utils.data_manager import DataLoader, DataSaver


with open('../cache/foreground_model.pkl', 'rb') as f:
    model, threshold = cPickle.load(f)

    x = T.matrix('x')
    y = model.get_output(x)
    output_func = theano.function(inputs=[x], outputs=(y >= threshold))

def _mask_dataset(dataset_filename, mask_filename):
    image_data = DataLoader(dataset_filename, verbose=True)

    # Pre-processing
    print "Pre-processing ..."

    images = image_data.get_all_images()
    for i, image in enumerate(images):
        image = imageproc.imresize(image, (160, 80, 3))
        image = imageproc.subtract_luminance(image)
        image = imageproc.scale_per_channel(image, [0, 1])
        images[i] = image

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
                    view_masks[k] = mask * orig_image
                    cur_index += 1

                mask_data.set_images(gid, pid, vid, view_masks)

    mask_data.save(mask_filename)

if __name__ == '__main__':
    _mask_dataset('../data/cuhk_large_detected.mat', '../data/cuhk_large_detected_mask.mat')
