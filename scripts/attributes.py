#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
import cPickle
import numpy
from scipy.io import loadmat
from reid.datasets import Datasets
from reid.preproc import imageproc
from reid.preproc.augment import aug_translation
from reid.models.layers import FullConnLayer, ConvPoolLayer
from reid.models.neural_net import NeuralNet, AutoEncoder
from reid.models import active_functions as actfuncs
from reid.models import cost_functions as costfuncs
from reid.optimization import sgd


attribute_names = ["accessoryFaceMask", "accessoryHairBand", "accessoryHat", "accessoryHeadphone", "accessoryKerchief", "accessoryMuffler", "accessoryNothing", "accessorySunglasses", "carryingBabyBuggy", "carryingBackpack", "carryingFolder", "carryingLuggageCase", "carryingMessengerBag", "carryingNothing", "carryingOther", "carryingPlasticBags", "carryingSuitcase", "carryingUmbrella", "footwearBlack", "footwearBlue", "footwearBoots", "footwearBrown", "footwearGreen", "footwearGrey", "footwearLeatherShoes", "footwearOrange", "footwearPink", "footwearRed", "footwearSandals", "footwearShoes", "footwearSneakers", "footwearStocking", "footwearWhite", "footwearYellow", "hairBald", "hairBlack", "hairBrown", "hairGrey", "hairLong", "hairOrange", "hairRed", "hairShort", "hairWhite", "hairYellow", "lowerBodyBlack", "lowerBodyBlue", "lowerBodyBrown", "lowerBodyCapri", "lowerBodyCasual", "lowerBodyFormal", "lowerBodyGreen", "lowerBodyGrey", "lowerBodyJeans", "lowerBodyLongSkirt", "lowerBodyOrange", "lowerBodyPink", "lowerBodyPlaid", "lowerBodyPurple", "lowerBodyRed", "lowerBodyShortSkirt", "lowerBodyShorts", "lowerBodySuits", "lowerBodyThickStripes", "lowerBodyTrousers", "lowerBodyWhite", "lowerBodyYellow", "personalFemale", "personalLarger60", "personalLess15", "personalLess30", "personalLess45", "personalLess60", "personalMale", "upperBodyBlack", "upperBodyBlue", "upperBodyBrown", "upperBodyCasual", "upperBodyFormal", "upperBodyGreen", "upperBodyGrey", "upperBodyJacket", "upperBodyLogo", "upperBodyLongSleeve", "upperBodyNoSleeve", "upperBodyOrange", "upperBodyOther", "upperBodyPink", "upperBodyPlaid", "upperBodyPurple", "upperBodyRed", "upperBodyShortSleeve", "upperBodySuit", "upperBodySweater", "upperBodyThickStripes", "upperBodyThinStripes", "upperBodyTshirt", "upperBodyVNeck", "upperBodyWhite", "upperBodyYellow"]

unique_groups = [
    ["hairBald", "hairLong", "hairShort"],
    ["hairBlack", "hairBrown", "hairGrey", "hairOrange", "hairRed", "hairWhite", "hairYellow"],
    ["personalFemale", "personalMale"],
    ["personalLarger60", "personalLess15", "personalLess30", "personalLess45", "personalLess60"]
]

multi_groups = [
    ["accessoryFaceMask", "accessoryHairBand", "accessoryHat", "accessoryHeadphone", "accessoryKerchief", "accessoryMuffler", "accessoryNothing", "accessorySunglasses"],
    ["carryingBabyBuggy", "carryingBackpack", "carryingFolder", "carryingLuggageCase", "carryingMessengerBag", "carryingNothing", "carryingOther", "carryingPlasticBags", "carryingSuitcase", "carryingUmbrella"],
    ["footwearBlack", "footwearBlue", "footwearBrown", "footwearGreen", "footwearGrey", "footwearOrange", "footwearPink", "footwearRed", "footwearWhite", "footwearYellow"],
    ["footwearBoots", "footwearLeatherShoes", "footwearSandals", "footwearShoes", "footwearSneakers", "footwearStocking"],
    ["lowerBodyBlack", "lowerBodyBlue", "lowerBodyBrown", "lowerBodyGreen", "lowerBodyGrey", "lowerBodyOrange", "lowerBodyPink", "lowerBodyPurple", "lowerBodyRed", "lowerBodyWhite", "lowerBodyYellow"],
    ["lowerBodyCapri", "lowerBodyCasual", "lowerBodyFormal", "lowerBodyJeans", "lowerBodyLongSkirt", "lowerBodyPlaid", "lowerBodyShortSkirt", "lowerBodyShorts", "lowerBodySuits", "lowerBodyThickStripes", "lowerBodyTrousers"],
    ["upperBodyBlack", "upperBodyBlue", "upperBodyBrown", "upperBodyGreen", "upperBodyGrey", "upperBodyOrange", "upperBodyPink", "upperBodyPurple", "upperBodyRed", "upperBodyWhite", "upperBodyYellow"],
    ["upperBodyCasual", "upperBodyFormal", "upperBodyJacket", "upperBodyLogo", "upperBodyLongSleeve", "upperBodyNoSleeve", "upperBodyOther", "upperBodyPlaid", "upperBodyShortSleeve", "upperBodySuit", "upperBodySweater", "upperBodyThickStripes", "upperBodyThinStripes", "upperBodyTshirt", "upperBodyVNeck"]
]

_cached_datasets = os.path.join('..', 'cache', 'attributes_datasets.pkl')
_cached_augment = os.path.join('..', 'cache', 'attributes_augment.pkl')
_cached_preproc = os.path.join('..', 'cache', 'attributes_preproc.pkl')
_cached_group = os.path.join('..', 'cache', 'attributes_group.pkl')
_cached_nndata = os.path.join('..', 'cache', 'attributes_nndata.pkl')
_cached_model = os.path.join('..', 'cache', 'attributes_model.pkl')

def _load_datasets(dataset_names, load_from_cache=False, save_to_cache=False):
    print "Loading Datasets ..."
    print "===================="

    if load_from_cache:
        with open(_cached_datasets, 'rb') as f:
            images, attributes = cPickle.load(f)
    else:
        images, attributes = [], []
        for dname in dataset_names:
            print "Loading {0}".format(dname)
            fname = os.path.join('..', 'data', 'attributes', dname+'.mat')
            tmp = loadmat(fname)
            dimages = tmp['images']
            dattributes = tmp['attributes']
            for i in xrange(dimages.shape[0]):
                for j in xrange(dimages.shape[1]):
                    if dimages[i, j].size == 0: break
                    images.append(dimages[i, j])
                    attributes.append(dattributes[i, 0].ravel())

    if save_to_cache:
        with open(_cached_datasets, 'wb') as f:
            cPickle.dump((images, attributes), f, protocol=cPickle.HIGHEST_PROTOCOL)

    return (images, attributes)

def _augment(images, attributes, load_from_cache=False, save_to_cache=False):
    print "Augmenting ..."
    print "=============="

    if load_from_cache:
        with open(_cached_augment, 'rb') as f:
            images, attributes = cPickle.load(f)
    else:
        images, attributes = aug_translation(images, attributes)

    if save_to_cache:
        with open(_cached_augment, 'wb') as f:
            cPickle.dump((images, attributes), f, protocol=cPickle.HIGHEST_PROTOCOL)

    return (images, attributes)

def _preproc(images, attributes, load_from_cache=False, save_to_cache=False):
    print "Preprocessing ..."
    print "================="

    if load_from_cache:
        with open(_cached_preproc, 'rb') as f:
            images, attributes = cPickle.load(f)
    else:
        for i, image in enumerate(images):
            image = imageproc.imresize(image, (80, 40), keep_ratio='height')
            image = imageproc.subtract_luminance(image)
            image = numpy.rollaxis(image, 2)
            images[i] = numpy.tanh(image)

    if save_to_cache:
        with open(_cached_preproc, 'wb') as f:
            cPickle.dump((images, attributes), f, protocol=cPickle.HIGHEST_PROTOCOL)

    return (images, attributes)

def _select_group(images, attributes, group, gtype, load_from_cache=False, save_to_cache=False):
    print "Selecting Group ..."
    print "==================="

    if load_from_cache:
        with open(_cached_group, 'rb') as f:
            selected_images, selected_attributes = cPickle.load(f)
    else:
        attr_id = [attribute_names.index(name) for name in group]
        attributes = [attr[attr_id] for attr in attributes]

        if gtype == 'unique':
            judge_func = lambda x: x.sum() == 1
            select_func = lambda x: numpy.where(x == 1)[0]
        else:
            judge_func = lambda x: x.sum() != 0
            select_func = lambda x: x

        selected_images = []
        selected_attributes = []

        for img, attr in zip(images, attributes):
            if judge_func(attr):  # Some data are mis-labeled
                selected_images.append(img)
                selected_attributes.append(select_func(attr))

        selected_images = imageproc.images2mat(selected_images)
        selected_attributes = imageproc.images2mat(selected_attributes)

    if save_to_cache:
        with open(_cached_group, 'wb') as f:
            cPickle.dump((selected_images, selected_attributes), f,
                protocol=cPickle.HIGHEST_PROTOCOL)

    return (selected_images, selected_attributes)

def _form_nndata(images, attributes, load_from_cache=False, save_to_cache=False):
    print "Forming NN-data ..."
    print "==================="

    if load_from_cache:
        with open(_cached_nndata, 'rb') as f:
            nndata = cPickle.load(f)
    else:
        nndata = Datasets(images, attributes)
        nndata.split(0.8, 0.1)

    if save_to_cache:
        with open(_cached_nndata, 'wb') as f:
            cPickle.dump(nndata, f, protocol=cPickle.HIGHEST_PROTOCOL)

    return nndata

def _train_model(nndata, load_from_cache=False, save_to_cache=True):
    print "Training Model ..."
    print "=================="

    if load_from_cache:
        with open(_cached_model, 'rb') as f:
            model = cPickle.load(f)
    else:
        layers = [ConvPoolLayer((20,3,5,5), (2,2), (3,80,40), actfuncs.tanh, False),
                  ConvPoolLayer((50,20,5,5), (2,2), None, actfuncs.tanh, True),
                  FullConnLayer(5950, 500, actfuncs.tanh),
                  FullConnLayer(500, 2, actfuncs.softmax)]

        model = NeuralNet(layers)

        sgd.train(model, nndata,
                  costfuncs.mean_binary_cross_entropy,
                  costfuncs.mean_zeroone_error_rate,
                  regularize=1e-2,
                  batch_size=500, n_epoch=200,
                  learning_rate=1e-1, momentum=0.9,
                  learning_rate_decr=0.95)

    if save_to_cache:
        with open(_cached_model, 'wb') as f:
            cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)

    return model

if __name__ == '__main__':
    dataset_names = [
        'prid',
        'CUHK',
        'MIT',
        'ViPER',
        '3DPeS',
        'i-Lid',
        'TownCentre',
        'GRID',
        'CAVIAR4REID',
        'SARC3D'
    ]

    images, attributes = _load_datasets(dataset_names, True, False)
    # images, attributes = _augment(images, attributes, True, False)

    images, attributes = _preproc(images, attributes, True, False)
    images, attributes = _select_group(images, attributes, unique_groups[2], 'unique', True, False)

    nndata = _form_nndata(images, attributes, True, False)

    model = _train_model(nndata, False, True)

