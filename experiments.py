#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:38:17 2019

@author: grollins
"""

import csv_processors

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

def allDrugsBinary():
    # TESTSET___ALLDRUGS   
            
    X0 = csv_processors.pandasFloatReader('0nMTreatmentDistalTraining.csv')
    X0t = csv_processors.pandasFloatReader('0nMTreatmentDistalTesting.csv')
    XAll = csv_processors.pandasFloatReader('AllTreatedtrainingdistal.csv')
    XAllt = csv_processors.pandasFloatReader('AllTreatedtestingdistal.csv')
    
    rawtraindata = [X0, XAll]
    rawtestdata = [X0t, XAllt]
    trainingkey = [0, 1]
    testingkey = [0, 1]
    
    trainPack = buildArrays(rawtraindata,trainingkey)
    testPack = buildArrays(rawtestdata, testingkey)
    
    return trainPack, testPack

def amiodarone():
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
    
    trainPack = buildArrays(rawtraindata,trainingkey)
    testPack = buildArrays(rawtestdata, testingkey)
    
    print(trainPack[0])
    
    return trainPack, testPack

def bortozemib():
    # TESTSET___Bortozemib
    
    
    # opening up a testing set
    X0 = csv_processors.pandasFloatReader('Bort0uMDistalTraining.csv')
    X100 = csv_processors.pandasFloatReader('Bort100nMDistalTraining.csv')
    X0test = csv_processors.pandasFloatReader('Bort0uMDistalTesting.csv')
    X100test = csv_processors.pandasFloatReader('Bort100nMDistalTesting.csv')
    
    rawtraindata = [X0, X100]
    rawtestdata = [X0test, X100test]
    
    trainingkey = [0, 100]
    testingkey = [0, 100]    
    
    trainPack = buildArrays(rawtraindata, trainingkey)
    testPack = buildArrays(rawtestdata, testingkey)
    
    
    return trainPack, testPack

def allDrugs5Cat():
    # TESTSET___allDrugs5Cat
    
    Xa = csv_processors.pandasFloatReader('amiotrainingdistal.csv')
    XaT = csv_processors.pandasFloatReader('amiotestingdistal.csv')
    Xv = csv_processors.pandasFloatReader('vincritrainingdistal.csv')
    XvT = csv_processors.pandasFloatReader('vincritestingdistal.csv')
    Xp = csv_processors.pandasFloatReader('paclitrainingdistal.csv')
    XpT = csv_processors.pandasFloatReader('paclitestingdistal.csv')
    Xo = csv_processors.pandasFloatReader('oxalitrainingdistal.csv')
    XoT = csv_processors.pandasFloatReader('oxalitestingdistal.csv')
    Xb = csv_processors.pandasFloatReader('borttrainingdistal.csv')
    XbT = csv_processors.pandasFloatReader('borttestingdistal.csv')
    
    rawtraindata = [Xa, Xv, Xp, Xo, Xb]
    rawtestdata = [XaT, XvT, XpT, XoT, XbT]
    
    trainingkey = [1,2,3,4,5]
    testingkey = [1,2,3,4,5]
    
    trainPack = buildArrays(rawtraindata, trainingkey)
    testPack = buildArrays(rawtestdata, testingkey)
    
    print(trainPack[0])
    
    return trainPack, testPack
