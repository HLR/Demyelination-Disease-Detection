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
from openpyxl import Workbook
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

def prepLine(switches, rtuple):
    for i in rtuple:
        switches.append(i)
    return switches

def addLine(line, sheet):
    rowmax = sheet.max_row+1
    for i in range(len(line)):
        sheet.cell(column = i + 1, row = rowmax, value = line[i])
# this initializes our Excel workbook to output
printoutwb = Workbook()
# initializing our Excel sheets to output
ws0 = printoutwb.create_sheet("All Data Dumped")
ws1 = printoutwb.create_sheet("1 Feature")
ws2 = printoutwb.create_sheet("2 Features")
ws3 = printoutwb.create_sheet("3 Features")
ws4 = printoutwb.create_sheet("4 Features")
ws5 = printoutwb.create_sheet("5 Features")
ws6 = printoutwb.create_sheet("6 Features")




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


for sheet in printoutwb:
    for col in range(1, len(randomFeatures)+ 1):
        sheet.cell(column= col, row= 1, value = randomFeatures[col-1].__name__)
printoutwb.save('printout.xlsx')



#create machine learning models
svmModel = svm.SVC()
rfcModel = ensemble.RandomForestClassifier()
mlpModel = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

#scaling our data seems to improve the SVC's results
scaler = StandardScaler()

test1= []
test2= []
test3= []
test4= []
test5= []
test6= []

for d0 in [0,1]:
    for d1 in [0,1]:
        for d2 in [0,1]:
            for d3 in [0,1]:
                for d4 in [0,1]:
                    for d5 in [0,1]:
                        dim = [d0,d1,d2,d3,d4,d5]
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
                            
                            if len(testList) == 1:
                                test1.append(plugTuple)
                                print(prepLine(dim, plugTuple))
                                addLine(prepLine(dim, plugTuple), ws1)
                            elif len(testList) == 2:
                                test2.append(plugTuple)
                                addLine(prepLine(dim, plugTuple), ws2)
                            elif len(testList) == 3:
                                test3.append(plugTuple)
                                addLine(prepLine(dim, plugTuple), ws3)
                            elif len(testList) == 4:
                                test4.append(plugTuple)
                                addLine(prepLine(dim, plugTuple), ws4)
                            elif len(testList) == 5:
                                test5.append(plugTuple)
                                addLine(prepLine(dim, plugTuple), ws5)
                            elif len(testList) == 6:
                                test6.append(plugTuple)
                                addLine(prepLine(dim, plugTuple), ws6)
                            

printoutwb.save('printout.xlsx')