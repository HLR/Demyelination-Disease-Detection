#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:07:17 2019

This file is to demonstrate how pickle works to save and store models for
later use. It will be configurable to be easy for Shees to use.

@author: grollins
"""

import pickle
import features
import experiments

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
trainPack, testPack, numcats = experiments.allDrugsBinary()
featureList = [features.min_value, features.max_value_i]
filename = 'finalized_model.sav'

trainfeatures = features.feature_loader(trainPack[0], featureList)
testfeatures = features.feature_loader(testPack[0], featureList)

model.fit(trainfeatures, trainPack[1])
print(model.score(testfeatures, testPack[1]))
pickle.dump(model, open(filename, 'wb'))

loadmodel = pickle.load(open(filename, 'rb'))
print(loadmodel.score(testfeatures, testPack[1]))