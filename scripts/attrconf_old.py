#!/usr/bin/python2
# -*- coding: utf-8 -*-

# datasets = ['3DPeS', 'CAVIAR4REID', 'CUHK', 'GRID', 'MIT', 'PRID', 'SARC3D', 'TownCentre', 'VIPeR', 'i-LIDS']
datasets = ['CAVIAR4REID', 'CUHK', 'GRID', 'PRID', 'TownCentre', 'VIPeR', 'i-LIDS']

names = ['accessoryFaceMask', 'accessoryHairBand', 'accessoryHat', 'accessoryHeadphone', 'accessoryKerchief', 'accessoryMuffler', 'accessoryNothing', 'accessorySunglasses', 'carryingBabyBuggy', 'carryingBackpack', 'carryingFolder', 'carryingLuggageCase', 'carryingMessengerBag', 'carryingNothing', 'carryingOther', 'carryingPlasticBags', 'carryingSuitcase', 'carryingUmbrella', 'footwearBlack', 'footwearBlue', 'footwearBoots', 'footwearBrown', 'footwearGreen', 'footwearGrey', 'footwearLeatherShoes', 'footwearOrange', 'footwearPink', 'footwearRed', 'footwearSandals', 'footwearShoes', 'footwearSneakers', 'lowerBodyStocking', 'footwearWhite', 'footwearYellow', 'hairBald', 'hairBlack', 'hairBrown', 'hairGrey', 'hairLong', 'hairOrange', 'hairRed', 'hairShort', 'hairWhite', 'hairYellow', 'lowerBodyBlack', 'lowerBodyBlue', 'lowerBodyBrown', 'lowerBodyCapri', 'lowerBodyCasual', 'lowerBodyFormal', 'lowerBodyGreen', 'lowerBodyGrey', 'lowerBodyJeans', 'lowerBodyLongSkirt', 'lowerBodyOrange', 'lowerBodyPink', 'lowerBodyPlaid', 'lowerBodyPurple', 'lowerBodyRed', 'lowerBodyShortSkirt', 'lowerBodyShorts', 'lowerBodySuits', 'lowerBodyThickStripes', 'lowerBodyTrousers', 'lowerBodyWhite', 'lowerBodyYellow', 'personalFemale', 'personalLarger60', 'personalLess15', 'personalLess30', 'personalLess45', 'personalLess60', 'personalMale', 'upperBodyBlack', 'upperBodyBlue', 'upperBodyBrown', 'upperBodyCasual', 'upperBodyFormal', 'upperBodyGreen', 'upperBodyGrey', 'upperBodyJacket', 'upperBodyLogo', 'upperBodyLongSleeve', 'upperBodyNoSleeve', 'upperBodyOrange', 'upperBodyOther', 'upperBodyPink', 'upperBodyPlaid', 'upperBodyPurple', 'upperBodyRed', 'upperBodyShortSleeve', 'upperBodySuit', 'upperBodySweater', 'upperBodyThickStripes', 'upperBodyThinStripes', 'upperBodyTshirt', 'upperBodyVNeck', 'upperBodyWhite', 'upperBodyYellow']

unival = [
    ['hairBlack', 'hairBrown', 'hairGrey', 'hairOrange', 'hairRed', 'hairWhite', 'hairYellow'],
    ['hairBald', 'hairLong', 'hairShort'],
    ['footwearBlack', 'footwearBlue', 'footwearBrown', 'footwearGreen', 'footwearGrey', 'footwearOrange', 'footwearPink', 'footwearRed', 'footwearWhite', 'footwearYellow'],
    ['footwearBoots', 'footwearLeatherShoes', 'footwearSandals', 'footwearShoes', 'footwearSneakers'],
    ['lowerBodyBlack', 'lowerBodyBlue', 'lowerBodyBrown', 'lowerBodyGreen', 'lowerBodyGrey', 'lowerBodyOrange', 'lowerBodyPink', 'lowerBodyPurple', 'lowerBodyRed', 'lowerBodyWhite', 'lowerBodyYellow'],
    ['personalFemale', 'personalMale'],
    ['personalLess15', 'personalLess30', 'personalLess45', 'personalLess60', 'personalLarger60']
]

unival_titles = [
    'Hair Color',
    'Hair Style',
    'Footwear Color',
    'Footwear Style',
    'Lower Body Color',
    'Gender',
    'Age'
]

multival = [
    ['accessoryFaceMask', 'accessoryHairBand', 'accessoryHat', 'accessoryHeadphone', 'accessoryKerchief', 'accessoryMuffler', 'accessoryNothing', 'accessorySunglasses'],
    ['carryingBabyBuggy', 'carryingBackpack', 'carryingFolder', 'carryingLuggageCase', 'carryingMessengerBag', 'carryingNothing', 'carryingOther', 'carryingPlasticBags', 'carryingSuitcase', 'carryingUmbrella'],
    ['lowerBodyCapri', 'lowerBodyCasual', 'lowerBodyFormal', 'lowerBodyJeans', 'lowerBodyLongSkirt', 'lowerBodyPlaid', 'lowerBodyShortSkirt', 'lowerBodyShorts', 'lowerBodySuits', 'lowerBodyThickStripes', 'lowerBodyTrousers', 'lowerBodyStocking'],
    ['upperBodyBlack', 'upperBodyBlue', 'upperBodyBrown', 'upperBodyGreen', 'upperBodyGrey', 'upperBodyOrange', 'upperBodyPink', 'upperBodyPurple', 'upperBodyRed', 'upperBodyWhite', 'upperBodyYellow'],
    ['upperBodyCasual', 'upperBodyFormal', 'upperBodyJacket', 'upperBodyLogo', 'upperBodyLongSleeve', 'upperBodyNoSleeve', 'upperBodyOther', 'upperBodyPlaid', 'upperBodyShortSleeve', 'upperBodySuit', 'upperBodySweater', 'upperBodyThickStripes', 'upperBodyThinStripes', 'upperBodyTshirt', 'upperBodyVNeck']
]

multival_titles = [
    'Accessories',
    'Carryings',
    'Lower Body Styles',
    'Upper Body Colors',
    'Upper Body Styles'
]
