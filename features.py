#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 11:30:36 2019

@author: grollins
"""

import numpy as np
import statistics as stats

def feature_loader(array, featurelist):
    """
    Here is what I want from this function:
        -I want to be able pass it
            1) an array that the function will extract features from
            2) the list of features to extract
        -I want to do it in such a way that I don't have to construct
        a massive case by case tree to grab the features. getattr() seems
        like a function that I may use to accomplish this.
    """
    returnX = []
    
    feature_key= []
    for i in range(len(featurelist)):
        feature_key.append(featurelist[i](array))
    
    for i in range(len(array)):
        addSample = []
        for j in range(len(feature_key)):
            addSample.append(feature_key[j][i])
        returnX.append(addSample)
        
    return returnX
        
def max_value(array):
    return np.amax(array, axis = 1)

def min_value(array):
    return np.amin(array, axis = 1)

def max_value_i(array):
    return np.argmax(array, axis = 1)

def min_value_i(array):
    return np.argmin(array, axis = 1)

def absolute_refractory(array):
    max_time = max_value_i(array)
    min_time = min_value_i(array)
    
    returnX = []
    for i in range(len(array)):
        returnX.append(min_time[i] - max_time[i])
        
    return returnX

def resting_potential_before_ap(array):
    """
    #TODO:
    This function will
        1) look at the max value index
        2) go back in time some certain amount
        3) select a large sequence of values
        4) return the median/ average of this sequence
    
    A sequel function will do the same but for after the action potential
    """
    max_time = max_value_i(array)
    returnX = []
    for x in range(len(max_time)):
        print(array[x][max_time[x]- 75: max_time[x] -50])
        returnX.append(stats.median(array[x][max_time[x]- 75: max_time[x] -50]))
    return returnX