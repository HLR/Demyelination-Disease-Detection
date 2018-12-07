#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 3rd

@author: grollins and Joseph Kim
"""

"""
Notes about the csv files!
    Each row is an array, with the first element being something like
    "Wave0" followed by comma separated values for that wave0.
    The Targets are manually set, and eventually a target file will be created
    for each corresponding data file.
"""

from sklearn import svm
import csv
import os
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import math
import random
from sklearn.externals import joblib
import csv_processors

def known_data_conjoiners(arrays, respective_categories):
    returnX = []
    returnY = []
    print(len(arrays))
    for i in range(len(arrays)):
        imax = np.amax(arrays[i], axis = 1)
        imin = np.amin(arrays[i], axis = 1)
        imaxi = np.argmax(arrays[i], axis = 1)
        imini = np.argmin(arrays[i], axis = 1)
        for j in range(len(arrays[i])):
            addPoint = []
            addPoint.append(imax[j])
            addPoint.append(imin[j])
            addPoint.append(imaxi[j])
            addPoint.append(imini[j])
            returnX.append(addPoint)
            returnY.append(respective_categories[i])
    
    return (returnX,returnY)


X0 = csv_processors.codyReader('Wave10bort0nmDistaltraining.csv')
X100 = csv_processors.codyReader('Wave10bort100nmDistaltraining.csv')
X0test = csv_processors.codyReader('Wave10bort0nmDistaltesting.csv')
X100test = csv_processors.codyReader('Wave10bort100nmDistaltesting.csv')

testPack= known_data_conjoiners([X0, X100], [0,100])
trainPack = known_data_conjoiners((X0test, X100test), (0,100))


svmModel = svm.SVC()
svmModel.fit(trainPack[0],trainPack[1])

for i in range(len(testPack[1])):
    print(testPack[1][i],"was the answer and the guess was",svmModel.predict([testPack[0][i]]))
print(svmModel.score(testPack[0],testPack[1]))

