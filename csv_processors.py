#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 19:20:27 2018

@author: gscollins, johykim, Shees Ahmed

Purpose: This file contains functions that read in our csv files for processing.
"""

import os
import pandas
import numpy as np

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
data_folder_path = os.path.join(script_dir, "Data Files")
# ^ "Data Files" is the folder we keep all our training csvs in
        
def pandasFloatReader(fileName):
    """
    The major file-reading function of our repository at the moment. This
    ignores wave label columns and takes the rest as 2D floats
    """
    
    with open(os.path.join(data_folder_path, fileName), 'r') as f:
        #creates the pandas csv_reader-generated DataFrame
        reader = pandas.read_csv(f, sep = None, engine = 'python', header = None)
        reader = reader.values
    rows1= np.array(reader)
    returner = rows1[:,1:len(rows1[0])-2]
    return returner.astype(float).tolist()
