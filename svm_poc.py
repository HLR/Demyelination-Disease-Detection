#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:02:49 2018

@author: grollins
"""

"""
Notes from Grant:
    after much much fiddling I have finally gotten the hang of the
    training.csv file, if not implemented that understanding.
    Each row is an array, with the first element being something like
    "Wave0" followed by comma separated values for that wave0.
    THE LAST ELEMENT OF EACH ROW IS ITS CATEGORY, OR TARGET VALUE

"""


#this will hopefully be the file to be used to show that we can run an svm
#thus import svm
from sklearn import svm

# we import python's library to handle csv files
# NOTING that apparently the pandas library is a better library to
# potentially use in the future
# it is found online
import csv

# I used the os import to allow us to fiddle with the directories
# on my computer, this works. Please let me know if it does not work on yours
# otherwise, I like this to help us keep our github organized
import os

# this is used below to help slice the arrays in such a way to separate
# quantitative data from strings and categories
import numpy as np

# These two lines are to help navigate our directories
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
data_folder_path = os.path.join(script_dir, "Data Files")

"""
This block below will take in the training file and read its data
but leave it very unprocessed in the array rawrows"""
with open(os.path.join(data_folder_path, "Training.csv"), 'r', ) as source:
    reader = csv.reader(source)
    rawrows = [row for row in reader]

"""This block partially processes the data
The resulting workrows array comes out as a 2D array, where each row contains
a string representing the wave #, following by string representations of the
x-values, finally followed by the training category as the last index"""
workrows =[]
for i in rawrows:
    addrow= []
    for j in i:
        addrow.append(j.strip())
    if len(addrow) != 0: workrows.append(addrow)

"""This is the block that uses numpy (as np) to slice our arrays in a way we
want.
Reminder:
    X is the numerical data. In the iris dataset it is the sepal length
    and width and such.
    In our csv file X is represented by all of the values of a row that are
    not the first or the last
    Y is the categorical data. In the iris dataset this represents the type
    of iris.
    """
workrows = np.array(workrows)
X = workrows[:, 1:-1]
y = workrows[:, -1]

"""
print(X) #prints a 2D array of all the data values for all the waves
print(y) #prints all the categories of all the waves
"""

"""Here is our proof of concept!!!"""
svmModel = svm.SVC()
svmModel.fit(X,y)
print(svmModel.score(X,y)) #This shows that if our data is trained on itself
# it can accurately diagnose itself!