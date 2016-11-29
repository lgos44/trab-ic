#!/usr/bin/python

import numpy as np
import csv
import sys, getopt
import matplotlib.pyplot as plt
import random
from scipy import stats
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
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

def splitDataset(dataset, target, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	data_train = []
	target_train = []
	copy = list(dataset)
	tcopy = list(target)
	while len(data_train) < trainSize:
		index = random.randrange(len(copy))
		data_train.append(copy.pop(index))
		target_train.append(tcopy.pop(index))
	return data_train, copy, target_train, tcopy

def statistics(data, individual=None):
	inds = []
	for i in range(0,10):
		inds.append(i)
		inds.append(i+10)
		inds.append(i+20)
	stats = []
	maxs = []
	mins = []
	mean = []
	std = []
	p25 = []
	p50 = []
	p75 = []
	for col in np.transpose(data):
		maxs.append(np.amax(col))
		mins.append(np.amin(col))
		mean.append(np.mean(col))
		std.append(np.std(col))	
		p25.append(np.percentile(col,25))
		p50.append(np.percentile(col,50))
		p75.append(np.percentile(col,75))
	stats = [maxs, mins, mean, std, p25, p50, p75]
	np.savetxt("mydata.csv", np.transpose(np.array(stats))[inds], delimiter=',')
	#print np.transpose(np.array(stats))[inds]

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
	ctrans = np.transpose(nparray)
	ctranspad = stats.zscore(ctrans)
	cpad = np.transpose(ctranspad)
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
		if y[i] != y_pred[i]: mislabeled = mislabeled + 1
		if y[i] == 1 and y_pred[i] == 1:
			vp = vp + 1
		elif y[i] == -1 and y_pred[i] == 1:
			fp = fp + 1
		elif y[i] == 1 and y_pred[i] == -1:	
			fn = fn + 1
		elif y[i] == -1 and y_pred[i] == -1:	
			vn = vn + 1
	print "VP = ", vp
	print "FP = ", fp
	print "FN = ", fn
	print "VN = ", vn 
	print "mislabeled :", mislabeled
	return [vp, fp, fn, vn]

def bayesianClassifier(data_train, data_test, target_train, target_test):
	gnb = GaussianNB()
	y_pred = gnb.fit(data_train, target_train).predict(data_test)
	return classifierStats(target_test, y_pred)

def quadraticClassifier(data_train, data_test, target_train, target_test):
	clf = QuadraticDiscriminantAnalysis()
	clf.fit(data_train, target_train)
	y_pred = clf.predict(data_test)
 	return classifierStats(target_test,y_pred)
	
def logisticRegression(data_train, data_test, target_train, target_test):
	logreg = linear_model.LogisticRegression(C=1e5)
	logreg.fit(data_train, target_train)
	Z = logreg.predict(data_test)
	return classifierStats(target_test,Z)

def multiLayerPerceptron(data_train, data_test, target_train, target_test):
	clf = MLPClassifier(solver='lbfgs', activation='tanh', max_iter=1000, hidden_layer_sizes=(100,100))
	clf.fit(data_train, target_train)	
	y_pred = clf.predict(data_test)
	return classifierStats(target_test,y_pred)

## Distance matrix and outlier removal
def distanceMatOut():
	#sort by class 1st column
	inds = target_np.argsort()
	target_np = target_np[inds[::1]]
	dataset_np = dataset_np[inds[::1]]
	#dataset_np = dataset_np[dataset_np[:,1].argsort()] NO
	dataset_zs = zscore(dataset_np)
	dist = distanceMatrix(dataset_zs)
	dataset_clean, target_clean = removeOutliersByDistance(dataset_zs, target_np, dist, 0.1)
	dist_clean = distanceMatrix(dataset_clean)
	return dataset_clean, target_clean


def main():
	np.set_printoptions(suppress=True)
	data, target = loadCsv('data2.csv')
	#statistics(data)
	splitratio = 0.80

#	dataset_clean, target_clean = distanceMatOut();
	bayes_simple = []
	logreg = []
	bayes_quad = []
	for i in range(0,1):
		print "MLP"
		data_train, data_test, target_train, target_test = splitDataset(data, target, splitratio)
		multiLayerPerceptron(data_train, data_test, target_train, target_test)
		print "Bayesian Classifier"
		stats = bayesianClassifier(data_train, data_test, target_train, target_test)
		bayes_simple.append(stats)
		print "LogisticRegression"
		stats = logisticRegression(data_train, data_test, target_train, target_test)
		logreg.append(stats)
		print "Quadratic Bayesian Classifier"
		stats = quadraticClassifier(data_train, data_test, target_train, target_test)
		bayes_quad.append(stats)
	
	print "Bayesian Classifier"	
	print np.mean(np.transpose(np.array(bayes_simple)), axis=1)
	print "LogisticRegression"
	print np.mean(np.transpose(np.array(logreg)), axis=1)
	print "Quadratic Bayesian Classifier"
	print np.mean(np.transpose(np.array(bayes_quad)), axis=1)
	
if __name__ == "__main__":
    main()
