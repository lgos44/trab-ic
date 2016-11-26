#!/usr/bin/python

import numpy as np
import csv
import sys, getopt
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model

variable_names = ['Diag', 'Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean', 'Smoothness Mean', 'Compactness Mean', 'Concavity Mean', 'Concave Points Mean',    'Symmetry Mean', 'Fractal Dimension Mean',	'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE', 'Smoothness SE', 'Compactness SE', 'Concavity SE',     'Concave points SE', 'Symmetry SE', 'Fractal Dimension SE',	'Radius Worst',	'Texture Worst', 'Perimeter Worst', 'Area Worst', 'Smoothness Worst',     'Compactness Worst',	'Concavity Worst', 'Concave Points Worst',	'Symmetry Worst', 'Fractal Dimension Worst']

# Loads csv discarting the first line (variable names) 
def loadCsv(filename):
	data = []
	target = []
	with open(filename, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		i = 0
		for row in reader:
			if i == 0: 
				i += 1
				continue
			data.append([float(row[j]) for j in range(1, len(row))])
			target.append(int(row[0]))
	return data, target

def correlationCoeffMatrix(filename):
	dataset = loadCsv(filename)
	dataset_np = np.array(dataset)
	dataset_np_transp = np.transpose(dataset_np)
	correlation_matrix = np.corrcoef(dataset_np_transp)
	v = np.linspace(-1.0, 1.0, 20, endpoint=True)
	plt.imshow(cm,interpolation='nearest', vmin=-1, vmax=1)
	plt.colorbar()
	plt.show()

# zscore of the nparray 
def zscore(nparray):
	#saves first column (output var)
	#matrix=np.asmatrix(nparray)
	#output=matrix[:,0]
	#utput_np=np.asarray(output)

	# Calculate zscore without output var
	#c = np.delete(nparray,0,1)
	ctrans = np.transpose(nparray)
	ctranspad = stats.zscore(ctrans)
	cpad = np.transpose(ctranspad)

	# Add output var again
	#data_zscore = np.concatenate((output_np, cpad), axis=1)
	return cpad
	
# euclidean distance matrix of a np array
def distanceMatrix(data):
	dim = np.shape(data)
	dist = np.zeros((dim[0],dim[0]))
	for i in range(0, dim[0]):
		for j in range(0, dim[0]):
			dist[i][j] = np.linalg.norm(data[i]-data[j])

	plt.imshow(dist,interpolation='nearest',origin='lower')
	plt.colorbar()
	plt.show()
	return dist

# Returns outliers indexes
def removeOutliersByDistance(data, target, dist, pout):
	## Distance mean
	mi = np.mean(dist, axis=1)
	mi_tuple = []
	index = 0
	for value in mi:
		mi_tuple.append((index,value))
		index += 1

	## Sorting by distance 
	mi_tuple.sort(key=lambda tup: tup[1])
	numreg = len(mi_tuple)*(1-pout)

	outliers = []
	for index in range(int(numreg),len(mi_tuple)):
		outliers.append(mi_tuple[index][0])

	data_clean = np.delete(data, (outliers), axis=0)
	target_clean = np.delete(target, (outliers), axis=0)
	return data_clean, target_clean

def classifierStats(y, y_pred):
	mislabeled = 0
	vp = 0
	fn = 0
	fp = 0
	vn = 0
	for i in range(0,np.shape(y_pred)[0]):
		if y[i] == 1 and y_pred[i] == 1:
			vp = vp + 1
		elif y[i] == -1 and y_pred[i] == 1:
			fp = fp + 1
		elif y[i] == 1 and y_pred[i] == -1:	
			fn = fn + 1
		elif y[i] == -1 and y_pred[i] == -1:	
			vn = vn + 1
	#print "Number of mislabeled points out of a total", np.shape(data)[0],  "points :", mislabeled
	print "VP = ", vp
	print "FP = ", fp
	print "FN = ", fn
	print "VN = ", vn 

def bayesianClassifier(data, target):
	gnb = GaussianNB()
	#data = np.delete(dataset_clean,0,1)
	#matrix=np.asmatrix(dataset_clean)
	#output=matrix[:,0]
	#target=np.asarray(output) 
	y_pred = gnb.fit(data, target).predict(data)
	classifierStats(target, y_pred)

def LogisticRegression(data, target):
	logreg = linear_model.LogisticRegression(C=1e5)
	logreg.fit(data, target)
	#h = .02
	#x_min, x_max = data[:, 0].min() - .5, data[:, 0].max() + .5
	#y_min, y_max = data[:, 1].min() - .5, data[:, 1].max() + .5
	#xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	#Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = logreg.predict(data)
	classifierStats(target,Z)


## Distance matrix and outline removal
data, target = loadCsv('data2.csv')

dataset_np = np.array(data)
target_np = np.array(target)
#sort by class 1st column
#np.set_printoptions(threshold=np.nan)
inds = target_np.argsort()
target_np = target_np[inds[::1]]
dataset_np = dataset_np[inds[::1]]
#dataset_np = dataset_np[dataset_np[:,1].argsort()] NO
dataset_zs = zscore(dataset_np)
dist = distanceMatrix(dataset_zs)
dataset_clean, target_clean = removeOutliersByDistance(dataset_zs, target_np, dist, 0.1)
dist_clean = distanceMatrix(dataset_clean)

bayesianClassifier(dataset_clean, target_clean)
LogisticRegression(dataset_clean, target_clean)



