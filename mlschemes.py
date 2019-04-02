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
from sklearn.model_selection import cross_val_score

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
    scores = cross_val_score(svmModel, testing_samples, testing_categories,\
                             cv=10)
    
    return [svmModel.score(testing_samples, testing_categories), prfs, scores]

def rfc(training_samples, training_categories, \
        testing_samples, testing_categories):
    
    rfcModel.fit(training_samples, training_categories)
    
    decisions = rfcModel.predict(testing_samples)
    prfs = precision_recall_fscore_support(testing_categories, decisions)
    scores = cross_val_score(rfcModel, testing_samples, testing_categories,\
                             cv=10)
    
    return [rfcModel.score(testing_samples, testing_categories), prfs, scores]
    
def mlp(training_samples, training_categories, \
        testing_samples, testing_categories):
    
    mlpModel.fit(training_samples, training_categories)
    
    decisions = mlpModel.predict(testing_samples)
    prfs = precision_recall_fscore_support(testing_categories, decisions)
    scores = cross_val_score(mlpModel, testing_samples, testing_categories,\
                             cv=10)
    
    return [mlpModel.score(testing_samples, testing_categories), prfs, scores]

def knn(training_samples, training_categories, \
        testing_samples, testing_categories):
    
    knnModel.fit(training_samples, training_categories)
    
    decisions = knnModel.predict(testing_samples)
    prfs = precision_recall_fscore_support(testing_categories, decisions)
    scores = cross_val_score(knnModel, testing_samples, testing_categories,\
                             cv=10)
    
    return [knnModel.score(testing_samples, testing_categories), prfs, scores]

def gnb(training_samples, training_categories, \
        testing_samples, testing_categories):
    
    gnbModel.fit(training_samples, training_categories)
    
    decisions = gnbModel.predict(testing_samples)
    prfs = precision_recall_fscore_support(testing_categories, decisions)
    scores = cross_val_score(gnbModel, testing_samples, testing_categories,\
                             cv=10)
    
    return [gnbModel.score(testing_samples, testing_categories), prfs, scores]

'''
    commented out for debugging purposes:
    svm, knn, gnb
'''
modelList = [rfc, mlp]