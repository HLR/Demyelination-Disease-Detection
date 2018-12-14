#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 3rd
@author: grollins, Joseph Kim, and Shees Ahmed
"""

'''
This file is different than the original svm_attempt.py file 
because this has a Graphical User Interface implemented in it 
that does not rely on local files. 
'''

"""
Notes about the csv files!
    Each row is an array, with the first element being something like
    "Wave0" followed by comma separated values for that wave0.
    The Targets are manually set, and eventually a target file will be created
    for each corresponding data file.
    
In order to run this code properly, the testing and training files must be in a 
"Data Files" folder and must be run with csv_processors.py for the data processing 
of the files. 
"""
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import ensemble
import csv
import os
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import math
import random
from sklearn.externals import joblib
import csv_processors

root = Tk()


def mfileopen():
    x = filedialog.askopenfilename()
    return x


def do_it():
    return filedialog.askopenfilename()





def known_data_conjoiners(arrays, respective_categories):
    """
    This function takes in arrays, a set of arrays, each containing
    one target category of data,
    as well as respective_categories, an array whose values at an indices
    represent arrays' category at the same index

    This function returns a tuple that easily pops into many machine learning
    functions provided by scikit
        returnX= the concatenated sample list
        returnY= the generated target list
    """

    # initialize returned arrays
    returnX = []
    returnY = []

    # build said arrays by taking all samples from arrays--the input,
    # extracting relevant features, and then jamming that into return X
    # At the end of processing our sample, throw its category into return Y
    for i in range(len(arrays)):
        imax = np.amax(arrays[i], axis=1)
        imin = np.amin(arrays[i], axis=1)
        imaxi = np.argmax(arrays[i], axis=1)
        imini = np.argmin(arrays[i], axis=1)

        for j in range(len(arrays[i])):
            addSample = []
            addSample.append(imax[j])
            addSample.append(imin[j])
            addSample.append(imaxi[j])
            addSample.append(imini[j])
            returnX.append(addSample)
            returnY.append(respective_categories[i])

    return [returnX, returnY]


root.title("Machine Learning Models")
root.geometry("640x640+0+0")
heading = Label(root, text="Machine Learning Models for DDD", font=("arial", 20, "bold"), fg="steelblue").pack()


def train():
    '''
    The train() function is the main part of the GUI that runs the training and testing of the models.
    It is executed when the button "Choose two training files and two testing files: " is selected on the
    GUI.
    '''

    '''
    The next three functions defined here, printSVM(), printRFC(), and printMLP(), allow for the 
    scoring to be outputted by the GUI on the interface itself. Once the two training and two testing files
    have been selected, the code automatically runs the models and then three buttons appear at the bottom of the 
    GUI. After pressing each button, the score of that particular model will be shown. 
    '''
    def printSVM():
        label = Label(root, text=svmModel.score(testPack[0], testPack[1]))
        label.place(x=300, y=500)

    def printRFC():
        label = Label(root, text=rfcModel.score(testPack[0], testPack[1]))
        label.place(x=300, y=550)

    def printMLP():
        label = Label(root, text=mlpModel.score(testPack[0], testPack[1]))
        label.place(x=300, y=600)

    # opening up a testing set
    '''
    When the button on the GUI is clicked, four 'file asks' are pulled up one at a time. The first two 
    prompts are for the testing files while the next two prompts are for the testing files.
    '''
    X0 = csv_processors.pandasFloatReader(mfileopen())#'Bort0nMDistalTraining.csv'
    X100 = csv_processors.pandasFloatReader(mfileopen()) #'Bort100nMDistalTraining.csv'
    X0test = csv_processors.pandasFloatReader(mfileopen())#'Bort0uMDistalTesting.csv'
    X100test = csv_processors.pandasFloatReader(mfileopen())#'Bort100nMDistalTesting.csv'

# set up our testing and training packages
    trainPack = known_data_conjoiners([X0, X100], [0, 100])
    testPack = known_data_conjoiners([X0test, X100test], [0, 100])

# scaling our data seems to improve the SVC's results
    scaler = StandardScaler()
    scaler.fit(trainPack[0])
    testPack[0] = scaler.transform(testPack[0])
    trainPack[0] = scaler.transform(trainPack[0])

    '''
    The model below is the Support Vector Machine model, or SVM. 
    '''
    svmModel = svm.SVC()
    svmModel.fit(trainPack[0], trainPack[1])
    button1 = Button(text="SVM Score: ", width=30, command=printSVM).place(x=10, y=500)
    print(svmModel.score(testPack[0], testPack[1]))

    '''
    The model below is the Random Forest Classifier.
    '''
    rfcModel = ensemble.RandomForestClassifier()
    rfcModel.fit(trainPack[0], trainPack[1])
    button2 = Button(text="RFC Score: ", width=30, command=printRFC).place(x=10, y=550)
    print(rfcModel.score(testPack[0], testPack[1]))

    '''
    The model below is the Multi-Layer Perceptron.
    '''
    mlpModel = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    mlpModel.fit(trainPack[0], trainPack[1])
    button3 = Button(text="MLP Score: ", width=30, command=printMLP).place(x=10, y=600)
    print(mlpModel.score(testPack[0], testPack[1]))


button9 = Button(text="Choose two training files and two testing files: ", width=60, command=train).place(x=10, y=200)

root.mainloop()
