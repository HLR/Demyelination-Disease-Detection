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

svmModel = svm.SVC()
rfcModel = ensemble.RandomForestClassifier()
mlpModel = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
knnModel = KNeighborsClassifier()
gnbModel = GaussianNB()


def svm(training_samples, training_categories, \
        testing_samples, testing_categories):
    
    svmModel.fit(training_samples, training_categories)
    return svmModel.score(testing_samples, testing_categories)

def rfc(training_samples, training_categories, \
        testing_samples, testing_categories):
    
    rfcModel.fit(training_samples, training_categories)
    return rfcModel.score(testing_samples, testing_categories)
    
def mlp(training_samples, training_categories, \
        testing_samples, testing_categories):
    
    mlpModel.fit(training_samples, training_categories)
    return mlpModel.score(testing_samples, testing_categories)

def knn(training_samples, training_categories, \
        testing_samples, testing_categories):
    
    knnModel.fit(training_samples, training_categories)
    return knnModel.score(testing_samples, testing_categories)

def gnb(training_samples, training_categories, \
        testing_samples, testing_categories):
    
    gnbModel.fit(training_samples, training_categories)
    return gnbModel.score(testing_samples, testing_categories)

modelList = [svm, rfc, mlp, knn, gnb]