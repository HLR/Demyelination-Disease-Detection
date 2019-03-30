#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 14:36:42 2019

@author: grollins
"""
import features

'''
this function is the second one you will be using in your server
You first load up an experiment from experiments.py
    that will return the values of trainPack and testPack
    which are passed into this function
Then, you will pass the list of machine name  abbreviations that the user wants
(see mlistparse for the key) as strings, which this function will then 
associate with the real machines.
The same will passed but for features

Notes!
--Right now, I can't find versions of random forest or k-nearest neighbor
classifiers that support incremental fitting, so they will not be an option,
unless you can find an option online that works
--the order of the 'if' tree in mlistparse is the same order that data points
will be returned to you in terms of which index corresponds to which machine

params:
    trainPack- directly passed as received from call an experiment
    testPack- same
    mlnames- string array of ml abbreviations. see mlistparse
    featurelist- string array of feature abbreviations. see flistparse
returns:
    mlist- a list of machine objects. see mlistparse
    flist - a list of features-dot functions that have been used on trainf
            and test f
    trainf - feature-extracted versions of samples in trainPack[0]
    testf- feature-extracted versions of samples in testPack[1]
'''
def sheesFunc0(trainPack, testPack, mlnames = ['mlp'],\
              featurelist = ['volt']):
    
    mlist = mlistparse(mlnames)
    flist = flistparse(featurelist)
    trainf = features.feature_loader(trainPack[0], flist)
    testf = features.feature_loader(testPack[0], flist)
    
    return [mlist, flist, trainf, testf]

'''
This function partially fits a new sample to a list of machines and returns
the new corresponding scores of those machines on the test set

params:
    sample- a sample tuple
        [0] - trainf[i] for a given i. 
        [1] - the target value of the given sample
    mlist- the list of machine instances that we are training
    t - a tuple containing the dedicated test set
        [0] - an array of feature-extracted samples
                should be equal to testf from earlier
        [1] - an array of their corresponding target values
                should probably be equal to testPack[1]
returns:
    an array containing the corresponding ml's scores
'''
def sheesFunc1(sample, mlist, t):
    r = []
    for i in mlist:
        i.partial_fit(sample[0], sample[1])
        r.append(i.score(t[0], t[1]))
    return r

def mlistparse(mlschemes):
    r = []
    if 'mlp' in mlschemes:
        from sklearn.neural_network import MLPClassifier
        r.append(\
                        MLPClassifier(solver='lbfgs', alpha=1e-5,\
                                      hidden_layer_sizes=(15,),\
                                      random_state=1))
    if 'svc' in mlschemes:
        from sklearn import linear_model
        r.append(linear_model.SGDClassifier())
    if 'gnb' in mlschemes:
        from sklearn.naive_bayes import GaussianNB
        r.append(GaussianNB())
    '''
    if 'knn' in mlschemes:
        from sklearn.neighbors import KNeighborsClassifier
        r.append(KNeighborsClassifier())
    if 'rfc' in mlschemes:
        from sklearn import ensemble
        r.append(ensemble.RandomForestClassifier())
    '''
    return r
        
def flistparse(fl):
    r = []
    if 'volt' in fl:
        r.append(features.volt)
    if 'mav' in fl:
        r.append(features.max_value)
    if 'mavi' in fl:
        r.append(features.max_value_i)
    if 'miv' in fl:
        r.append(features.min_value)
    if 'mivi' in fl:
        r.append(features.min_value_i)
    if 'ar' in fl:
        r.append(features.absolute_refractory)
    return r