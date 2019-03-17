#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 11:30:36 2019

@author: grollins
"""

import numpy as np
import statistics as stats


"""
This feature will take in an array and a list of features to be extracted
and it returns an array of the initial array's length by num of features
"""
def feature_loader(array, featurelist):
    """
    Here is what I want from this function:
        -I want to be able pass it
            1) an array that the function will extract features from
            2) the list of features to extract
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

"""
This helper function grabs only the numerical data points from a wave function
"""
def wavegrab(array):
    rows1= np.array(array)
    returner = rows1[:,1:len(rows1[0])-2]
    return returner.astype(float).tolist()
        
def volt(array):
    wavestrings = np.array(array)[:,0]
    wavefloat = []
    for i in range(len(wavestrings)):
        wavefloat.append(float(wavestrings[i][4:]))
    return wavefloat

def max_value(array):
    wave = wavegrab(array)
    return np.nanmax(wave, axis = 1)

def min_value(array):
    wave = wavegrab(array)
    return np.nanmin(wave, axis = 1)

def max_value_i(array):
    wave = wavegrab(array)
    return np.argmax(wave, axis = 1)

def min_value_i(array):
    wave = wavegrab(array)
    return np.argmin(wave, axis = 1)

def absolute_refractory(array):
    wave = wavegrab(array)
    max_time = max_value_i(wave)
    min_time = min_value_i(wave)
    
    returnX = []
    for i in range(len(wave)):
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
    wave = wavegrab(array)
    
    max_time = max_value_i(wave)
    returnX = []
    for x in range(len(max_time)):
        # print(array[x][max_time[x]- 75: max_time[x] -50])
        returnX.append(stats.median(wave[x][max_time[x]- 75: max_time[x] -50]))
    return returnX

#commented out resting_potential_before_ap!
featureList = [max_value, min_value, max_value_i, min_value_i, \
               absolute_refractory, \
               volt]