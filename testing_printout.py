#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 17:02:29 2019

@author: grollins
"""

from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import ensemble
import csv
import numpy as np
import os
import features
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import math
import random
from sklearn.externals import joblib
import csv_processors

def dataPrep(arrays, respective_categories, featureList):
    """
    This function takes in arrays, a set of arrays, each containing
    one target category of data,
    as well as respective_categories, an array whose values at an indices
    represent arrays' category at the same index
    
    This function returns the arrays conjoined into a single array, returnX
    along with a conjoined list of their respective categories
    """
    
    returnX = []
    returnY = []
    
    for i in range(len(arrays)):
        for j in range(len(arrays[i])):
            returnX.append(arrays[i][j])
            returnY.append(respective_categories[i])
            
    returnX = features.feature_loader(returnX, featureList)
    return [returnX, returnY]



# opening up a testing set
X0 = csv_processors.pandasFloatReader('Bort0uMDistalTraining.csv')
X100 = csv_processors.pandasFloatReader('Bort100nMDistalTraining.csv')
X0test = csv_processors.pandasFloatReader('Bort0uMDistalTesting.csv')
X100test = csv_processors.pandasFloatReader('Bort100nMDistalTesting.csv')

# this ndarray will store the values of every machine learning
# iteration run on every possible combination of 6 randomly selected
# features
testMaster = np.ndarray(shape = (2,2,2,2,2,2,3))
randomFeatures = np.random.choice(features.featureList, 6, replace = False)

#scaling our data seems to improve the SVC's results
scaler = StandardScaler()
svmModel = svm.SVC()
rfcModel = ensemble.RandomForestClassifier()
mlpModel = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

for d0 in [0,1]:
    for d1 in [0,1]:
        for d2 in [0,1]:
            for d3 in [0,1]:
                for d4 in [0,1]:
                    for d5 in [0,1]:
                        testList = []
                        if d0 == 1:
                            testList.append(randomFeatures[0])
                        if d1 == 1:
                            testList.append(randomFeatures[1])
                        if d2 == 1:
                            testList.append(randomFeatures[2])
                        if d3 == 1:
                            testList.append(randomFeatures[3])
                        if d4 == 1:
                            testList.append(randomFeatures[4])
                        if d5 == 1:
                            testList.append(randomFeatures[5])
                        
                        if len(testList) == 0:
                            testMaster[d0][d1][d2][d3][d4][d5] = [0.0,0.0,0.0]
                        else:
                            # set up our testing and training packages
                            trainPack= dataPrep([X0, X100], [0,100], testList)
                            testPack = dataPrep([X0test, X100test], [0,100], testList)
                            
                            scaler.fit(trainPack[0])
                            testPack[0] = scaler.transform(testPack[0])
                            trainPack[0] = scaler.transform(trainPack[0])
                            
                            svmModel.fit(trainPack[0],trainPack[1])
                            rfcModel.fit(trainPack[0],trainPack[1])
                            mlpModel.fit(trainPack[0],trainPack[1])
                            
                            plugTuple = [svmModel.score(testPack[0],testPack[1]), \
                                         rfcModel.score(testPack[0],testPack[1]), \
                                         mlpModel.score(testPack[0],testPack[1])]
                            
                            testMaster[d0][d1][d2][d3][d4][d5] = \
                            plugTuple
                            print(testMaster[d0][d1][d2][d3][d4][d5])
