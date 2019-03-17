#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 17:02:29 2019

@author: grollins
"""

# the following is a data scaler which improves performance
from sklearn.preprocessing import StandardScaler 

import experiments
import mlschemes
import features
import numpy as np
from sklearn.cross_validation import train_test_split
from openpyxl import Workbook
from autoencoderRBM import svm_on_BRBM


def prepLine(switches, rtuple):
    arrayLine = switches.copy()
    for i in rtuple:
        arrayLine.append(i)
    return arrayLine

def addLine(line, sheet):
    rowmax = sheet.max_row+1
    for i in range(len(line)):
        sheet.cell(column = i + 1, row = rowmax, value = line[i])

"""
configuration variables
"""
trainPack, testPack = experiments.amiodarone()
outputFileName = None
highscoreThreshold = .6

# this initializes our Excel workbook to output
printoutwb = Workbook()
# initializing our Excel sheets to output
ws1 = printoutwb.create_sheet("1 Feature")
ws2 = printoutwb.create_sheet("2 Features")
ws3 = printoutwb.create_sheet("3 Features")
ws4 = printoutwb.create_sheet("4 Features")
ws5 = printoutwb.create_sheet("5 Features")
ws6 = printoutwb.create_sheet("6 Features")
maxws = printoutwb.create_sheet("Highscores")
autows = printoutwb.create_sheet("Autoencoder")
printoutwb.remove(printoutwb['Sheet'])

# this randomly selects 6 of our designed features to test
# this is done because runtime of this file is O(2^n) where n is the 
# number of features tested
randomFeatures = np.random.choice(features.featureList, 6, replace = False)

# this writes the header for every sheet in excel
for i in range(len(printoutwb.worksheets)-1):
    keyLine = randomFeatures.copy()
    for j in range(len(keyLine)):
        keyLine[j] = keyLine[j].__name__
    keyLine = keyLine.tolist()
    keyLine.extend(['scheme' , '20% of training set' , '36%', '52%', '68%', '84%', '100%'])
    addLine(keyLine, printoutwb.worksheets[i])

#scaling our data seems to improve the SVC's results
scaler = StandardScaler()

for d0 in [0,1]:
    for d1 in [0,1]:
        for d2 in [0,1]:
            for d3 in [0,1]:
                for d4 in [0,1]:
                    for d5 in [0,1]:
                        dim = [d0,d1,d2,d3,d4,d5]
                        testList = []
                        learningcurves = np.zeros((len(mlschemes.modelList),6))
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
                            
                        if not testList:
                            pass
                        else:
                            
                            trainfeatures = features.feature_loader(trainPack[0],\
                                                                    testList)
                            testfeatures = features.feature_loader(testPack[0],\
                                                                   testList)
                            
                            
                            for i in range(5):
                                trainsplitsamples, _ , trainsplitcats, _ = \
                                train_test_split(trainfeatures, trainPack[1], \
                                                 test_size = (.2 + (i*.16)))
                
                                for j in range(len(mlschemes.modelList)):
                                    learningcurves[j][i] = \
                                    mlschemes.modelList[j](\
                                            trainsplitsamples,\
                                            trainsplitcats,\
                                            testfeatures,\
                                            testPack[1]
                                            )
                                
                            for j in range(len(mlschemes.modelList)):
                                learningcurves[j][5] = \
                                mlschemes.modelList[j](\
                                        trainfeatures,\
                                        trainPack[1],\
                                        testfeatures,\
                                        testPack[1]
                                        )
                                
                                plugTuple = [mlschemes.modelList[j].__name__]
                                plugTuple.extend(learningcurves[j])
                                
                                
                                # this is the printout block
                                if len(testList) == 1:
                                    addLine(prepLine(dim, plugTuple), ws1)
                                elif len(testList) == 2:
                                    addLine(prepLine(dim, plugTuple), ws2)
                                elif len(testList) == 3:
                                    addLine(prepLine(dim, plugTuple), ws3)
                                elif len(testList) == 4:
                                    addLine(prepLine(dim, plugTuple), ws4)
                                elif len(testList) == 5:
                                    addLine(prepLine(dim, plugTuple), ws5)
                                elif len(testList) == 6:
                                    addLine(prepLine(dim, plugTuple), ws6)
                                
                                if mlschemes.modelList[j](\
                                        trainfeatures,\
                                        trainPack[1],\
                                        testfeatures,\
                                        testPack[1]
                                        ) > highscoreThreshold:
                                    addLine(prepLine(dim,plugTuple), maxws)

printoutwb.save(outputFileName)            

for i in range(5):
    plugTuple = []
    trainsplitsamples, _ , trainsplitcats, _ = \
    train_test_split(trainPack[0], trainPack[1], \
                     test_size = (.2 + (i*.16)))
    returntuple = svm_on_BRBM(trainsplitsamples, trainsplitcats,\
                              testPack[0], testPack[1])
    plugTuple.append(.5 + (i*.1))
    plugTuple.append(returntuple[1])
    addLine(plugTuple, autows)
plugTuple = []
plugTuple.append(1)
plugTuple.append(svm_on_BRBM(trainPack[0], trainPack[1], testPack[0], testPack[1])[1])
addLine(plugTuple, autows)

printoutwb.save(outputFileName)