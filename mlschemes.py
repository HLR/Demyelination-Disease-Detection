#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 23:18:27 2019

This file holds all of our machine learning schemes.
The idea is to simplify our code by making function pipelines that fit, scale,
and score data in a black box

@author: grollins
"""

from sklearn import ensemble
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support

svmModel = svm.SVC()
rfcModel = ensemble.RandomForestClassifier()
mlpModel = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
knnModel = KNeighborsClassifier()
gnbModel = GaussianNB()


def svm(training_samples, training_categories, \
        testing_samples, testing_categories):
    
    svmModel.fit(training_samples, training_categories)
    
    decisions = svmModel.predict(testing_samples)
    prfs = precision_recall_fscore_support(testing_categories, decisions)
    
    return [svmModel.score(testing_samples, testing_categories), prfs]

def rfc(training_samples, training_categories, \
        testing_samples, testing_categories):
    
    rfcModel.fit(training_samples, training_categories)
    
    decisions = rfcModel.predict(testing_samples)
    prfs = precision_recall_fscore_support(testing_categories, decisions)
    
    return [rfcModel.score(testing_samples, testing_categories), prfs]
    
def mlp(training_samples, training_categories, \
        testing_samples, testing_categories):
    
    mlpModel.fit(training_samples, training_categories)
    
    decisions = mlpModel.predict(testing_samples)
    prfs = precision_recall_fscore_support(testing_categories, decisions)
    
    return [mlpModel.score(testing_samples, testing_categories), prfs]

def knn(training_samples, training_categories, \
        testing_samples, testing_categories):
    
    knnModel.fit(training_samples, training_categories)
    
    decisions = knnModel.predict(testing_samples)
    prfs = precision_recall_fscore_support(testing_categories, decisions)
    
    return [knnModel.score(testing_samples, testing_categories), prfs]

def gnb(training_samples, training_categories, \
        testing_samples, testing_categories):
    
    gnbModel.fit(training_samples, training_categories)
    
    decisions = gnbModel.predict(testing_samples)
    prfs = precision_recall_fscore_support(testing_categories, decisions)
    
    return [gnbModel.score(testing_samples, testing_categories), prfs]

'''
    commented out for debugging purposes:
    svm, knn, gnb
'''
modelList = [rfc, mlp]