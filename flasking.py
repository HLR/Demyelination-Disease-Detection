'''
@author: shees-a

To run this code on Windows, go to the command line, "cd" into the directory that this folder is placed in, and then run by typing in 
"python flasking.py"

This is the main body of code used to run the web interface. The backend is run by using "flask" which as many good 
features that help make the start up of the website straightforward while also allowing user input and running the 
backend logic to be smooth as well. However, the downside is the database management is harder to work with. 
'''




#So far, there have been no problems with any imports (although some may not be completely necessary)
from flask import Flask, render_template, jsonify, request
from flask import Markup
from werkzeug.utils import secure_filename
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
from experiments import allDrugsBinary, amiodarone, bortozemib, allDrugs5Cat, userDrugfile
from sklearn import model_selection
import pickle


#This is the standard way to instantiate a flask app and is necessary. 
app = Flask(__name__)


'''The '/' is the root directory, meaning that it is the home page. 
If a new webpage is to be added that requires user input, then there must be an
 '@app.route('/name', methods = ['GET', 'POST'])' line. 


 Once the route is made, there are some restrictions on what can be done with 
 the webpage. For instance, there can only be one function that runs per webpage
 and that one function must have a return statement that is not "None". But, 
 the function can return a template that needs to be rendered (hence, "render_template("name.html")")
 which actually displays the start page. 
'''
@app.route('/', methods = ['GET','POST'])
def upload_experiments():
   return render_template('experiment.html')

'''This is the last page that was made in the initial run of the Capstone 
project, and does not have the full functionality it is intended to. It can take 
in a user's file that they want to input but it cannot run that file with the saved 
model. This would be the next task to complete for further development of the webpage. 
'''
@app.route('/treat', methods = ['GET','POST'])
def uploading_experiments():
	if request.method == 'POST': 
		a = request.files['File']
		b = secure_filename(a.filename)
		c = userDrugfile(b)
		d = ["volt"]
		e = ["gnb"]
		Yval = []
		f = sheesFunc0(c, c, mlnames = e, featurelist = d)
		print("This is f: " + f)
		g = sheesFunc1(f[3], f[0], [f[3], c], d)
		print("This is g: " + g)

		return "This is the highest accuracy obtained with your file: " 



'''This is the main route and function that actually takes in the user input from 
'experiment.html' and runs the functions in gui_funcs.py. 
'''

@app.route('/uploader', methods = ['GET','POST'])
def upload(): 

	'''
	request.form[''] is the main command that takes in the user input from the experiments.py 
	template and places it in a, c, and d. 
	'''
	a = request.form['drug']
	c = request.form.getlist('feature')
	d = request.form.getlist('scheme')

	'''
	The choice that user can pick has been hardcoded into four if statements 
	since it became difficult to try to generalize the code. 
	'''

	if a == "Amiodarone":

	'''
	b is the call to experiments.py where the files are pre-loaded into the model. 
	Xval and Yval are going to be the lists that store the values of the iterations 
	and accuracy scores, respectively. 

	train and test are the arrays built from experiments.py that give the data from the loaded 
	in files. 
	''' 
		b = amiodarone()
		Xval = []
		Yval = []
		train = b[0]
		test = b[1]
		targets = b[2]

	'''
	The next variable is e, which stores in the result of sheesFunc0 (the details of this 
	function is described in gui_funcs.py)
	The value of e is stored in mlist, sample, testset, and t. 
	'''

		e = sheesFunc0(train, test, mlnames = d, featurelist = c)
		mlist = e[0]
		sample = e[2]
		testset = e[3]
		t = [e[3], test[1]]
	
	'''
	This is the block of code that actually produces the accuracy scores by 
	going through sheesFunc1 and produces the graph that plots the iterations 
	and accuracy scores. Again, the details of sheesFunc1 are in gui_funcs.py

	There was a strange phenomenon in this code where having certain print statements actually 
	allowed the code to run whereas if the print statements like 'print(f)' where not there then
	the code would crash. This did not happen consistently so if there is an error in running this 
	that is something to look into. 
	'''

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
	'''
	This block allows the plot to actually be made and any of the parameters can be adjusted to make the 
	graph appear different
	'''

			plt.plot(i, Yval[i], color='green', marker='.', linestyle='solid', linewidth=2, markersize=5)
			plt.ylabel('Accuracy Score')
			plt.xlabel('Number of Iterations')
			plt.title('Amiodarone')
			plt.pause(0.05)

		plt.show()

	'''
	After the graph is closed, then the html file 'newfile.html' is loaded. This is where a user would 
	input their own file. 
	'''
		return render_template('newfile.html')


	'''
	The next three if statements have the same format as the previously described code, so 
	the descriptions would be similar except for which drug files are loaded in. 
	'''

	if a == "Bortozemib": 
		b = bortozemib()
		Xval = []
		Yval = []
		train = b[0]
		test = b[1]
		targets = b[2]

		e = sheesFunc0(train, test, mlnames = d, featurelist = c)
		mlist = e[0]
		sample = e[2]
		testset = e[3]
		t = [e[3], test[1]]
		
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
			plt.ylabel('Accuracy Score')
			plt.xlabel('Number of Iterations')
			plt.title('Bortozemib')
			plt.pause(0.05)
			
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
			plt.ylabel('Accuracy Score')
			plt.xlabel('Number of Iterations')
			plt.title('All Drugs (Binary)')
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
	
		e = sheesFunc0(train, test, mlnames = d, featurelist = c)
		
		mlist = e[0]
		sample = e[2]
		testset = e[3]
		t = [e[3], test[1]]
	
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
			plt.ylabel('Accuracy Score')
			plt.xlabel('Number of Iterations')
			plt.title('All Drugs (5 Categories)')
			plt.pause(0.05)
			print('completed sheesFunc1')

		plt.show()
		return render_template('newfile.html')


'''
This is a necessary piece of code to run the website. 
Having debug=True allows for changing the code and 
reloading the website without having to stop the entire 
site and running from the command line again. 
'''
if __name__=='__main__':
	app.run(debug=True)