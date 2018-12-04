#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 3rd

@author: grollins and Joseph Kim
"""

"""
Notes about the csv files!
    Each row is an array, with the first element being something like
    "Wave0" followed by comma separated values for that wave0.
    The Targets are manually set, and eventually a target file will be created
    for each corresponding data file.
"""

from sklearn import svm
import csv
import os
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import math
import random
from sklearn.externals import joblib


with open('Wave10bort0nmDistaltraining.csv', 'r', ) as f:
	#creates the CSV reader class
	reader = csv.reader(f)
	#extracts rows for the file
	rows1 = [row for row in reader]
	#gets the first value* in each row
	column = [row[1] for row in reader]
	a=len(rows1)
rows1=np.array(rows1)

with open('Wave10bort100nmDistaltraining.csv', 'r', ) as f:
	reader1 = csv.reader(f)
	rows2 = [row for row in reader1]
	column1 = [row[1] for row in reader1]
	b=len(rows2)
rows2=np.array(rows2)

with open('Wave10bort0nmDistaltesting.csv', 'r', ) as f:
	reader2 = csv.reader(f)
	rows3 = [row for row in reader2]
	column2 = [row[1] for row in reader2]
	c=len(rows3)
rows3=np.array(rows3)

with open('Wave10bort100nmDistaltesting.csv', 'r', ) as f:
	reader3 = csv.reader(f)
	rows4 = [row for row in reader3]
	column3 = [row[1] for row in reader3]
	d=len(rows4)
rows4=np.array(rows4)


X0 = rows1[:,1:len(rows1[0])-2]
X100 = rows2[:,1:len(rows2[0])-2]
X0test = rows3[:,1:len(rows3[0])-2]
X100test = rows4[:,1:len(rows4[0])-2]

#y=rows1[:,len(rows1[0])-1]
print(len(X0))
print(len(X100))
print(len(X0test))
print(len(X100test))
X0=X0.astype(float)
X100 = X100.astype(float)
X0test=X0test.astype(float)
X100test = X100test.astype(float)
#y=y.astype(float)
ytrain = []
for i in range(len(X0)): ytrain.append(0)
for i in range(len(X100)): ytrain.append(100)

ytest = []
for i in range(len(X0test)): ytest.append(0)
for i in range(len(X100test)): ytest.append(100)

'''
#(possibly) remove noise from data
for i in range(len(X0)-1,-1,-1):
	if abs(float(np.max(X0[i]))) < 50:
		X0=np.delete(X0,i,axis=0)
print(len(X0))

for i in range(len(X100)-1,-1,-1):
	if abs(float(np.max(X100[i]))) < 50:
		X100=np.delete(X100,i,axis=0)
print(len(X100))

for i in range(len(X0test)-1,-1,-1):
	if abs(float(np.max(X0test[i]))) < 50:
		X0test=np.delete(X0test,i,axis=0)
print(len(X0test))

for i in range(len(X100test)-1,-1,-1):
	if abs(float(np.max(X100test[i]))) < 50:
		X100test=np.delete(X100test,i,axis=0)
print(len(X100test))
'''
max_X0 = np.amax(X0, axis=1)
max_X100 = np.amax(X100, axis=1)
max_X0test = np.amax(X0test, axis=1)
max_X100test = np.amax(X100test, axis=1)


min_X0 = np.amin(X0, axis=1)
min_X100 = np.amin(X100, axis=1)
min_X0test = np.amin(X0, axis=1)
min_X100test = np.amin(X100, axis=1)


index_max_X0 = np.argmax(X0, axis=1)
index_max_X100 = np.argmax(X100, axis=1)
index_min_X0 = np.argmin(X0, axis=1)
index_min_X100 = np.argmin(X100, axis=1)
index_max_X0test = np.argmax(X0, axis=1)
index_max_X100test = np.argmax(X100, axis=1)
index_min_X0test = np.argmin(X0, axis=1)
index_min_X100test = np.argmin(X100, axis=1)


xtrain_features = []
for i in range(len(X0)):
	buildarray=[]
	buildarray.append(max_X0[i])
	buildarray.append(min_X0[i])
	buildarray.append(index_max_X0[i])
	buildarray.append(index_min_X0[i])
	xtrain_features.append(buildarray)
for i in range(len(X100)):
	buildarray = []
	buildarray.append(max_X100[i])
	buildarray.append(min_X100[i])
	buildarray.append(index_max_X100[i])
	buildarray.append(index_min_X100[i])
	xtrain_features.append(buildarray)

print("\n\n\n")

xtest_features = []
for i in range(len(X0test)):
	buildarray=[]
	print(max_X0test[i])
	buildarray.append(max_X0test[i])
	buildarray.append(min_X0test[i])
	buildarray.append(index_max_X0test[i])
	buildarray.append(index_min_X0test[i])
	xtest_features.append(buildarray)
for i in range(len(X100test)):
	buildarray = []
	buildarray.append(max_X100test[i])
	buildarray.append(min_X100test[i])
	buildarray.append(index_max_X100test[i])
	buildarray.append(index_min_X100test[i])
	xtest_features.append(buildarray)

print(xtrain_features)

svmModel = svm.SVC()
svmModel.fit(xtrain_features, ytrain)

print(svmModel.score(xtest_features, ytest))

for i in range(0, 8, 1):
	print(ytest[i])
	print(svmModel.predict(xtest_features[i]))
	print("\n")

'''
a=list(range(len(X)))
slice = random.sample(a, len(a))
XX1=X[slice[1]]
yy1=y[slice[1]]
XX2=XX1
yy2=yy1
for j in range(2,len(slice)):
	if j<0.8*len(a):
		XX1=np.row_stack((XX1, X[slice[j]]))
		yy1=np.row_stack((yy1, y[slice[j]]))
	else:
		XX2=np.row_stack((XX2, X[slice[j]]))
		yy2=np.row_stack((yy2, y[slice[j]]))

print(len(XX1),len(XX1[0]))
print(len(yy1))

'''
