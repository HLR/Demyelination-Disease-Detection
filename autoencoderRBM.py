#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 14:07:46 2019

@author: grollins
"""
"""
This is a Bernoulli Restricted Boltzmann Machine
It is an autoencoder we use to compress features
"""
from sklearn.neural_network import BernoulliRBM
from sklearn import svm
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
        buildX = []
        buildX.append(float(arrays[i][0][4:]))
        buildX.extend(arrays[i][1:])
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
    for i in range(len(testing_arrays)):
        buildX = []
        buildX.append(float(testing_arrays[i][0][4:]))
        buildX.extend(testing_arrays[i][1:])
        testX.append(buildX)
        testY.append(testing_respective_categories[i])
    
    comptestX = rbm.transform(testX)
    
    return [learning_rate, classifier.score(comptestX, testY)]

