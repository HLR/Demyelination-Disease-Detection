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
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import ensemble
import features
import csv_processors

def known_data_conjoiners(arrays, respective_categories):
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
            
    featuresToTest = [features.absolute_refractory]
    returnX = features.feature_loader(returnX, featuresToTest)
    return [returnX, returnY]



# opening up a testing set
X0 = csv_processors.pandasFloatReader('Bort0uMDistalTraining.csv')
X100 = csv_processors.pandasFloatReader('Bort100nMDistalTraining.csv')
X0test = csv_processors.pandasFloatReader('Bort0uMDistalTesting.csv')
X100test = csv_processors.pandasFloatReader('Bort100nMDistalTesting.csv')


# set up our testing and training packages
trainPack= known_data_conjoiners([X0, X100], [0,100])
testPack = known_data_conjoiners([X0test, X100test], [0,100])

#scaling our data seems to improve the SVC's results
scaler = StandardScaler()
scaler.fit(trainPack[0])
testPack[0] = scaler.transform(testPack[0])
trainPack[0] = scaler.transform(trainPack[0])

svmModel = svm.SVC()
svmModel.fit(trainPack[0],trainPack[1])
print(svmModel.score(testPack[0],testPack[1]))

rfcModel = ensemble.RandomForestClassifier()
rfcModel.fit(trainPack[0], trainPack[1])
print(rfcModel.score(testPack[0],testPack[1]))

mlpModel = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
mlpModel.fit(trainPack[0], trainPack[1])
print(mlpModel.score(testPack[0],testPack[1]))