from flask import Flask, render_template, jsonify, request
from flask import Markup
#from werkzeug import secure_filenames
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
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv_processors
import features
from gui_funcs import sheesFunc0, sheesFunc1, mlistparse, flistparse
from experiments import allDrugsBinary, amiodarone, bortozemib, allDrugs5Cat



app = Flask(__name__)


@app.route('/', methods = ['GET','POST'])
def upload_experiments():
   return render_template('experiment.html')


@app.route('/treat', methods = ['GET','POST'])
def uploading_experiments():
   return render_template('treat.html')

@app.route('/uploader', methods = ['GET','POST'])
def upload(): 
	a = request.form['drug']
	c = request.form.getlist('feature')
	d = request.form.getlist('scheme')

	print('got all required info')

	if a == "Amiodarone": 
		b = amiodarone()
		Xval = []
		Yval = []
		train = b[0]
		test = b[1]
		targets = b[2]
		#hj = str(targets)
		#print("This is targets: " + hj)
		print('trying to go through sheesFunc')
		e = sheesFunc0(train, test, mlnames = d, featurelist = c)
		print('got thru sheesfunc0')
		mlist = e[0]
		sample = e[2]
		testset = e[3]
		t = [e[3], test[1]]
		print('attempting to iterate through loop')
		for i in range(len(sample)):
			Xval.append(i)
			builder = []
			builder.append(sample[i])
			builder.append(train[1][i])
			print(i)
			print('attempting sheesFunc1c')
			if i == 0:
				f = sheesFunc1(builder, mlist, t, targets)
				print(f)
				Yval.append(f[0])
			else:
				f = sheesFunc1(builder, mlist, t, targets)
				print(f)
				Yval.append(f[0])

			plt.plot(i, Yval[i], color='green', marker='.', linestyle='solid', linewidth=2, markersize=5)
			plt.pause(0.05)
			print('completed sheesFunc1')

		#plt.plot(Xval, Yval) 
  
# naming the x axis 
		#plt.xlabel('Number of Iterations') 
# naming the y axis 
		#plt.ylabel('Accuracy Score') 
  
# giving a title to my graph 
		#plt.title('Machine Learning Plot') 
  
# function to show the plot 
		plt.show()
		return render_template('newfile.html')

	if a == "Bortozemib": 
		b = bortozemib()
		Xval = []
		Yval = []
		train = b[0]
		test = b[1]
		targets = b[2]
		#hj = str(targets)
		#print("This is targets: " + hj)
		print('trying to go through sheesFunc')
		e = sheesFunc0(train, test, mlnames = d, featurelist = c)
		print('got thru sheesfunc0')
		mlist = e[0]
		sample = e[2]
		testset = e[3]
		t = [e[3], test[1]]
		print('attempting to iterate through loop')
		for i in range(len(sample)):
			Xval.append(i)
			builder = []
			builder.append(sample[i])
			builder.append(train[1][i])
			print(i)
			print('attempting sheesFunc1c')
			if i == 0:
				f = sheesFunc1(builder, mlist, t, targets)
				print(f)
				Yval.append(f[0])
			else:
				f = sheesFunc1(builder, mlist, t, targets)
				print(f)
				Yval.append(f[0])

			plt.plot(i, Yval[i], color='green', marker='.', linestyle='solid', linewidth=2, markersize=5)
			plt.pause(0.05)
			print('completed sheesFunc1')


		#plt.plot(Xval, Yval) 
  
# naming the x axis 
		#plt.xlabel('Number of Iterations') 
# naming the y axis 
		#plt.ylabel('Accuracy Score') 
  
# giving a title to my graph 
		#plt.title('Machine Learning Plot') 
  
# function to show the plot 
		plt.show() 

		return render_template('newfile.html')

	if a == "AllDrugsBi":
		b = allDrugsBinary()
		Xval = []
		Yval = []
		train = b[0]
		test = b[1]
		targets = b[2]
		#hj = str(targets)
		#print("This is targets: " + hj)
		print('trying to go through sheesFunc')
		e = sheesFunc0(train, test, mlnames = d, featurelist = c)
		print('got thru sheesfunc0')
		mlist = e[0]
		sample = e[2]
		testset = e[3]
		t = [e[3], test[1]]
		print('attempting to iterate through loop')
		for i in range(len(sample)):
			Xval.append(i)
			builder = []
			builder.append(sample[i])
			builder.append(train[1][i])
			print(i)
			print('attempting sheesFunc1c')
			if i == 0:
				f = sheesFunc1(builder, mlist, t, targets)
				print(f)
				Yval.append(f[0])
			else:
				f = sheesFunc1(builder, mlist, t, targets)
				print(f)
				Yval.append(f[0])

			plt.plot(i, Yval[i], color='green', marker='.', linestyle='solid', linewidth=2, markersize=5)
			plt.pause(0.05)
			print('completed sheesFunc1')

		#plt.plot(Xval, Yval) 
  
# naming the x axis 
		#plt.xlabel('Number of Iterations') 
# naming the y axis 
		#plt.ylabel('Accuracy Score') 
  
# giving a title to my graph 
		#plt.title('Machine Learning Plot') 
  
# function to show the plot 
		plt.show()
		return render_template('newfile.html')

	if a == "AllDrugs5":
		b = allDrugs5Cat()
		Xval = []
		Yval = []
		train = b[0]
		test = b[1]
		targets = b[2]
		#hj = str(targets)
		#print("This is targets: " + hj)
		print('trying to go through sheesFunc')
		e = sheesFunc0(train, test, mlnames = d, featurelist = c)
		print('got thru sheesfunc0')
		mlist = e[0]
		sample = e[2]
		testset = e[3]
		t = [e[3], test[1]]
		print('attempting to iterate through loop')
		for i in range(len(sample)):
			Xval.append(i)
			builder = []
			builder.append(sample[i])
			builder.append(train[1][i])
			print(i)
			print('attempting sheesFunc1c')
			if i == 0:
				f = sheesFunc1(builder, mlist, t, targets)
				print(f)
				Yval.append(f[0])
			else:
				f = sheesFunc1(builder, mlist, t, targets)
				print(f)
				Yval.append(f[0])

			plt.plot(i, Yval[i], color='green', marker='.', linestyle='solid', linewidth=2, markersize=5)
			plt.pause(0.05)
			print('completed sheesFunc1')

		#plt.plot(Xval, Yval) 
  
# naming the x axis 
		#plt.xlabel('Number of Iterations') 
# naming the y axis 
		#plt.ylabel('Accuracy Score') 
  
# giving a title to my graph 
		#plt.title('Machine Learning Plot') 
  
# function to show the plot 
		plt.show()
		return render_template('newfile.html')
 
if __name__=='__main__':
	app.run(debug=True)