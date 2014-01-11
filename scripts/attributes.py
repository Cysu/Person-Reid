#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
import cPickle
from scipy.io import loadmat
from reid.preproc.augment import aug_translation


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

def _load_datasets(dataset_names, load_from_cache=False, save_to_cache=False):
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
                    attributes.append(dattributes[i, 0])

    if save_to_cache:
        with open(_cached_datasets, 'wb') as f:
            cPickle.dump((images, attributes), f, protocol=cPickle.HIGHEST_PROTOCOL)

    return (images, attributes)

def _augment(images, attributes, load_from_cache=False, save_to_cache=False):
    if load_from_cache:
        with open(_cached_augment, 'rb') as f:
            images, attributes = cPickle.load(f)
    else:
        images, attributes = aug_translation(images, attributes)

    if save_to_cache:
        with open(_cached_augment, 'wb') as f:
            cPickle.dump((images, attributes), f, protocol=cPickle.HIGHEST_PROTOCOL)

    return (images, attributes)

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

    raw_images, raw_attributes = _load_datasets(dataset_names, False, True)
    aug_images, aug_attributes = _augment(raw_images, raw_attributes, False, True)
