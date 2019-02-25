#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 17:02:29 2019

@author: grollins
"""

"""
The following imports are all classifiers or scalers used for data
classification
"""
from sklearn import svm
from sklearn.preprocessing import StandardScaler 

"""
This is a Bernoulli Restricted Boltzmann Machine
It is an autoencoder we use to compress features
"""
from sklearn.neural_network import BernoulliRBM

import mlschemes
import csv_processors
import features
import numpy as np
from sklearn.cross_validation import train_test_split
from openpyxl import Workbook

def buildArrays(arrays, respective_categories):
    """
    params:
        arrays- a set of arrays,
        each of shape n_i by m with
                n = the number of samples
                n_i = the number of samples in this array
                m = the number of features per sample
        and each containing one target category of data
        respective_categories- an array whose values at an indices represent
        arrays' category at the same index
        
    returns:
        [X,y]:
            X- a set of arrays of shape n by m
            y- 1D array of target values at X[i]
    """
    X = []
    y = []
    
    for i in range(len(arrays)):
        for j in range(len(arrays[i])):
            X.append(arrays[i][j])
            y.append(respective_categories[i])
    
    return [X,y]

# this runs our full wave through a autoencoder to compress it
def bernoulliCompress(arrays, respective_categories, learning_rate):
    """
    params:
        arrays- a set of arrays, each containing one target category of data
        respective_categories- an array whose values at an indices represent
            arrays' category at the same index
        learning_rate- 'The learning rate for weight updates [of BRM.]
            It is highly recommended to tune this hyper-parameter.
            Reasonable values are in the 10**[0., -3.] range.'
    
    returns:
        package- 2 element array containing new_X, y
            new_X- numpy array of shape [n_samples, n_features_new]
            y- array containing target values for n samples
            rbm- the Bernoulli transformer
            
    Update: After hypertuning, it appears that learning rate does not really
    matter too much in these cases.
    .1 works as a good default unless further tuning is required.
    """
    
    X = []
    y = []
    
    #relocate this to improve runtime
    for i in range(len(arrays)):
        for j in range(len(arrays[i])):
            buildX = []
            buildX.append(float(arrays[i][j][0][4:]))
            buildX.extend(arrays[i][j][1:])
            X.append(buildX)
            y.append(respective_categories[i])
            
    rbm = BernoulliRBM(learning_rate = learning_rate)
    
    new_X = rbm.fit_transform(X, y)
    return [new_X, y, rbm]

def svm_on_BRBM(training_arrays, training_respective_categories, \
                testing_arrays, testing_respective_categories, \
                learning_rate = .1):
    """
    params:
        training_arrays- a set of arrays for training, each containing one 
            target category of data
        training_respective_categories- an array whose values at an indices 
            represent training_arrays' category at the same index
        testing_arrays- a set of arrays for testing, each containing one 
            target category of data
        testing_respective_categories- an array whose values at an indices 
            represent testing_arrays' category at the same index
        learning_rate- 'The learning rate for weight updates [of BRM.]
            It is highly recommended to tune this hyper-parameter.
            Reasonable values are in the 10**[0., -3.] range.'

    returns:
        line- an array consisting of [learning_rate, score(new_X,y)]
    """
    classifier = svm.SVC()
    compX, trainY, rbm = bernoulliCompress(training_arrays, \
                                     training_respective_categories,\
                                     learning_rate)
    classifier.fit(compX, trainY)
    
    testX = []
    testY = []
    for i in range(len(training_arrays)):
        for j in range(len(training_arrays[i])):
            buildX = []
            buildX.append(float(training_arrays[i][j][0][4:]))
            buildX.extend(training_arrays[i][j][1:])
            testX.append(buildX)
            testY.append(testing_respective_categories[i])
    
    comptestX = rbm.transform(testX)
    
    return [learning_rate, classifier.score(comptestX, testY)]



# def dataPrep(arrays, respective_categories, featureList):
    """
    This function takes in arrays, a set of arrays, each containing
    one target category of data,
    as well as respective_categories, an array whose values at an indices
    represent arrays' category at the same index
    
    This function returns the arrays conjoined into a single array, returnX
    along with a conjoined list of their respective categories
    """
    """
    returnX = []
    returnY = []
    
    for i in range(len(arrays)):
        for j in range(len(arrays[i])):
            returnX.append(arrays[i][j])
            returnY.append(respective_categories[i])
            
    returnX = features.feature_loader(returnX, featureList)
    return [returnX, returnY]
"""


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
# TESTSET___AMIODARONE



# this block loads in the relevant data
X01d = csv_processors.pandasFloatReader('amiodarone0.1uMdistaltraining.csv')
X01dtest = csv_processors.pandasFloatReader('amiodarone0.1uMdistaltesting.csv')
X01p = csv_processors.pandasFloatReader('amiodarone0.1uMproximaltraining.csv')
X01ptest = csv_processors.pandasFloatReader('amiodarone0.1uMproximaltesting.csv')
X0d = csv_processors.pandasFloatReader('amiodarone0uMdistaltraining.csv')
X0dtest = csv_processors.pandasFloatReader('amiodarone0uMdistaltesting.csv')
X0p = csv_processors.pandasFloatReader('amiodarone0uMproximaltraining.csv')
X0ptest = csv_processors.pandasFloatReader('amiodarone0uMproximaltesting.csv')
X1d = csv_processors.pandasFloatReader('amiodarone1uMdistaltraining.csv')
X1dtest = csv_processors.pandasFloatReader('amiodarone1uMdistaltesting.csv')
X1p = csv_processors.pandasFloatReader('amiodarone1uMproximaltraining.csv')
X1ptest = csv_processors.pandasFloatReader('amiodarone1uMproximaltesting.csv')
X5d = csv_processors.pandasFloatReader('amiodarone5uMdistaltraining.csv')
X5dtest = csv_processors.pandasFloatReader('amiodarone5uMdistaltesting.csv')
X5p = csv_processors.pandasFloatReader('amiodarone5uMproximaltraining.csv')
X5ptest = csv_processors.pandasFloatReader('amiodarone5uMproximaltesting.csv')
X10d = csv_processors.pandasFloatReader('amiodarone10uMdistaltraining.csv')
X10dtest = csv_processors.pandasFloatReader('amiodarone10uMdistaltesting.csv')
X10p = csv_processors.pandasFloatReader('amiodarone10uMproximaltraining.csv')
X10ptest = csv_processors.pandasFloatReader('amiodarone10uMproximaltesting.csv')

rawtraindata = [X01d, X01p, X0d, X0p, X1d, X1p, X5d, X5p, X10d, X10p]
rawtestdata = [X01dtest, X01ptest, X0dtest, X0ptest, X1dtest,\
               X1ptest, X5dtest, X5ptest, X10dtest, X10ptest]

# this measurement is in tenths of a uM
trainingkey = [1, 1, 0, 0, 10, 10, 50, 50, 100, 100]
testingkey = [1, 1, 0, 0, 10, 10, 50, 50, 100, 100]
"""

# TESTSET___Bortozemib


# opening up a testing set
X0 = csv_processors.pandasFloatReader('Bort0uMDistalTraining.csv')
X100 = csv_processors.pandasFloatReader('Bort100nMDistalTraining.csv')
X0test = csv_processors.pandasFloatReader('Bort0uMDistalTesting.csv')
X100test = csv_processors.pandasFloatReader('Bort100nMDistalTesting.csv')

rawtraindata = [X0, X100]
rawtestdata = [X0test, X100test]

trainingkey = [0,100]
testingkey = [0, 100]



# this initializes our Excel workbook to output
printoutwb = Workbook()
# initializing our Excel sheets to output
ws1 = printoutwb.create_sheet("1 Feature")
ws2 = printoutwb.create_sheet("2 Features")
ws3 = printoutwb.create_sheet("3 Features")
ws4 = printoutwb.create_sheet("4 Features")
ws5 = printoutwb.create_sheet("5 Features")
ws6 = printoutwb.create_sheet("6 Features")
printoutwb.remove(printoutwb['Sheet'])



# this randomly selects 6 of our designed features to test
# this is done because runtime of this file is O(2^n) where n is the 
# number of features tested
randomFeatures = np.random.choice(features.featureList, 6, replace = False)

# this writes the header for every sheet in excel
for sheet in printoutwb:
    keyLine = randomFeatures.copy()
    for i in range(len(keyLine)):
        keyLine[i] = keyLine[i].__name__
    keyLine = keyLine.tolist()
    keyLine.extend(['scheme' , '50% of training set' , '60%', '70%', '80%', '90%', '100%'])
    addLine(keyLine, sheet)


#scaling our data seems to improve the SVC's results
scaler = StandardScaler()


trainPack = buildArrays(rawtraindata, trainingkey)
testPack = buildArrays(rawtestdata, testingkey)


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
                                                 test_size = (.5 + (i*.1)))
                                
                
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
                                
                                plugTuple = []




printoutwb.save('learningcurvebort.xlsx')