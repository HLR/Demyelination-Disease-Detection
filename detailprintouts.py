#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:49:50 2019

@author: grollins
"""


# the following is a data scaler which improves performance
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix
import experiments
import mlschemes
import features
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier

trainPack, testPack, numcats = experiments.bortozemib()
# populate learning Splits with the fractions of the training sets that you
# would like to test. Always include 1.

#scaling our data seems to improve the SVC's results
scaler = StandardScaler()



"""
confusion matrix work!
"""
clf = svm.SVC()

flist = [features.min_value, features.min_value_i, features.max_value, \
         features.volt]

trainfeatures = features.feature_loader(trainPack[0], flist)
testfeatures = features.feature_loader(testPack[0], flist)

trainfeatures = scaler.fit_transform(trainfeatures)
testfeatures = scaler.transform(testfeatures)

clf.fit(trainfeatures, trainPack[1])

y_pred = clf.predict(testfeatures)
print(confusion_matrix(testPack[1], y_pred))

scores = cross_val_score(clf, testfeatures, testPack[1],\
                             cv=10)
print(scores.mean())
print(scores.std())

print('vs dummy')

dum = DummyClassifier(strategy = 'stratified')
dum.fit(trainfeatures, trainPack[1])
dum_pred = dum.predict(testfeatures)

print(confusion_matrix(testPack[1], dum_pred))

dumbscores = cross_val_score(dum, testfeatures, testPack[1],\
                             cv=10)
print(dumbscores.mean())
print(dumbscores.std())