#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:02:49 2018

@author: grollins
"""

"""
Notes about the csv files!
    Each row is an array, with the first element being something like
    "Wave0" followed by comma separated values for that wave0.
    THE LAST ELEMENT OF EACH ROW IS ITS CATEGORY, OR TARGET VALUE

"""

from sklearn import svm
import csv
import os
import numpy as np
from sklearn.datasets import load_iris

def read_in_array(fileName, numX = 100):
    """
    This function takes in a fileName string (including .csv suffix,)
    as well as a numX defining the number of X feature cells to be examined
    (simply from left to right, for example it defaults to taking the first
    100 leftmost x-data cells)
    It outputs a tuple containing this extracted @2D x-data array in the first index,
    and a 1 dimensional array representing the true target values in its second
    """
    
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    data_folder_path = os.path.join(script_dir, "Data Files")
    # ^ "Data Files" is the folder we keep all our training csvs in
    
    """
    This block below will take in the training file and read its data
    but leave it very unprocessed in the array rawrows
    """
    with open(os.path.join(data_folder_path, fileName), 'r', ) as source:
        reader = csv.reader(source)
        rawrows = [row for row in reader]

    """
    The resulting workrows array comes out as a 2D array, where each row contains
    a string representing the wave #, following by string representations of the
    x-values, finally followed by the training category as the last index"""
    workrows =[]
    for i in rawrows:
        addrow= []
        for j in i:
            addrow.append(j.strip())
        if len(addrow) != 0: workrows.append(addrow)

    """This is the block that uses numpy (as np) to slice our arrays
    based on the format of the CSV files. See notes at top of file for
    discussion of csv files' formatting.
    Reminder:
        X is the numerical data. In the iris dataset it is the sepal length
        and width and such.
        In our csv file X is represented by all of the values of a row that are
        not the first or the last
        Y is the categorical data. In the iris dataset this represents the type
        of iris.
    """
    workrows = np.array(workrows)
    X = workrows[:, 1: numX]
    # print(X)
    y = workrows[:, -1]
    # print(y)
    return (X, y)

training_tuple = read_in_array("Training.csv", 1950)

# Internal testing! Why is it always wrong when I plug in our dataset?
"""
The block below is supposed to train an svmModel on every element in the training
set besides one (our index i), then attempt to predict the category of that i.
It shows that the iris dataset is 97.3% accurate in this training scheme,
but 0% accurate on ours and I am curious as to why.

Comments are in quotes, but code adjustments to switch to our dataset are in
# comments.
"""
iris = load_iris()
#comment out when issues with my code are resolved

correct_counter = 0
for i in range(len(iris.target)):
    # replace iris.target with training_tuple[0]
    
    """Create a fresh SVC model per interation"""
    svmModel = svm.SVC()
    
    """Here we load all but one elements into training arrays"""
    # holderX = np.delete(training_tuple[0].copy(), i, 0)
    holderX = np.delete(iris.data.copy(), i, 0)
    # holderY = np.delete(training_tuple[1].copy(), i)
    holderY = np.delete(iris.target.copy(), i)
    
    """We train on these all-but-ones"""
    svmModel.fit(holderX, holderY)
    
    """We predict the but-one datapoint's category, then compare it to
    the answer, then tally"""
    #prediction_result = svmModel.predict([training_tuple[0][i]])
    prediction_result = svmModel.predict([iris.data[i]])
    # print(str(prediction_result) + "   " + str(i) + "   " + str(iris.target[i]))
    
    if prediction_result[0] == iris.target[i]:
        correct_counter = correct_counter + 1

"""Prints the ratio of correctness"""
print(correct_counter / len(iris.target))
# replace iris.target with training_tuple[0]
