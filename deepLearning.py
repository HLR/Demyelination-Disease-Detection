#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 12:18:21 2018

@author: grollins
"""

from csv_processors import codyReader
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPClassifier

xtrain = []
ytrain= []
for i in codyReader("Wave10bort0nmDistaltraining.csv"):
    xtrain.append(i)
    ytrain.append(0)
    
#uncomment the block below to prove to yourself that our deep learning model
#isn't just spitting out garbage guesses
"""for i in codyReader("Wave10bort0nmDistaltesting.csv"):
    xtrain.append(i)
    ytrain.append(0)"""
    
for i in codyReader("Wave10bort100nmDistaltraining.csv"):
    xtrain.append(i)
    ytrain.append(100)
    
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(15,), random_state=1)

scaler = StandardScaler()  
scaler.fit(xtrain)  
xtrain = scaler.transform(xtrain)

clf.fit(xtrain, ytrain)

xtest = []
ytest = []
for i in codyReader("Wave10bort0nmDistaltesting.csv"):
    xtest.append(i)
    ytest.append(0)
for i in codyReader("Wave10bort100nmDistaltesting.csv"):
    xtest.append(i)
    ytest.append(100)
    
xtest= scaler.transform(xtest)

print(clf.predict(xtest))
print(ytest)