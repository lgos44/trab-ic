#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import csv
import sys, getopt
import matplotlib.pyplot as plt
import random
from scipy import stats
from sklearn.preprocessing import StandardScaler 
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.decomposition import PCA

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

def kfoldSplit(dataset, target, cicles, index):
	size = int(len(dataset)/cicles)+1
	end = (index+1)*size
	if end > len(dataset): end = len(dataset)
	data_train = [dataset[i] for i in range(0,index*size)+range((index+1)*size,len(dataset))]
	data_test = [dataset[i] for i in range(index*size,  end)]
	target_train = [target[i] for i in range(0,index*size)+range((index+1)*size,len(dataset))]
	target_test = [target[i] for i in range(index*size,  end)]
	return data_train, data_test, target_train, target_test

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

def perceptron(data_train, data_test, target_train, target_test):
	prp = linear_model.Perceptron(penalty=None, class_weight='balanced')
	prp.fit(data_train, target_train)
	y_pred = prp.predict(data_test)
	return classifierStats(target_test,y_pred)

def multiLayerPerceptron(data_train, data_test, target_train, target_test):
	clf = MLPClassifier(solver='sgd', activation='tanh', max_iter=1000, hidden_layer_sizes=(21,))
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

def plotPREREC(confmat, name):
	width = 0.5
	ind1 = [1,1+width+.1]
	ind2 = [4,4+width+.1]
	ind = np.arange(7) 
	
	vp = confmat[0]
	fp = confmat[1]
	fn = confmat[2]
	vn = confmat[3]
	pre_c1 = vp/(vp+fp)
	pre_c2 = vn/(vn+fn)
	rec_c1 = vp/(vp+fn)
	rec_c2 = vn/(vn+fp)
	print pre_c1, rec_c1, pre_c2, rec_c2
	print 'AUC = ', (rec_c1+rec_c2)/2 
	print "ACC = ", (vp+vn)/(vp+fp+fn+vn) 
	p1 = plt.bar(ind1, [pre_c1,rec_c1], width, color='r')
	p2 = plt.bar(ind2, [pre_c2,rec_c2], width, color='b')
	plt.ylabel('PRE/REC')
	plt.title('Precisao e Recall')
	plt.yticks(np.arange(0, 1, 0.1))
	plt.legend((p1[0], p2[0]), ('C1', 'C2'))
	#plt.xticks(ind + width/2.)
	plt.xticks([0,(ind1[0]+ind1[1]+width)/2,(ind2[0]+ind2[1]+width)/2,6],('','C1','C2',''))
	plt.show()
	#plt.savefig(name)

def saveConfMat(confmat, name):
	np.savetxt(name, confmat.reshape(2,2,order='F'), fmt='%.2f', delimiter='&', newline='\\\\ \n')

def main():
	np.set_printoptions(suppress=True)
	data, target = loadCsv('data2.csv')
	#statistics(data)
	
	#dataset_clean, target_clean = distanceMatOut();
	bayes_simple = []
	logreg = []
	bayes_quad = []
	percep = []
	mlp = []
	

	for i in range(0,10):
		data_train, data_test, target_train, target_test = kfoldSplit(data, target, 10, i)
		#data_train = zscore(data_train)
		print "Bayesian Classifier"
		stats = bayesianClassifier(data_train, data_test, target_train, target_test)
		bayes_simple.append(stats)
		print "LogisticRegression"
		stats = logisticRegression(data_train, data_test, target_train, target_test)
		logreg.append(stats)
		print "Quadratic Bayesian Classifier"
		stats = quadraticClassifier(data_train, data_test, target_train, target_test)
		bayes_quad.append(stats)

		#data_test = zscore(data_test)
		scaler = StandardScaler()  
		scaler.fit(data_train)  
		data_train = scaler.transform(data_train)  
		data_test = scaler.transform(data_test)  
		print "Perceptron"
		stats = perceptron(data_train, data_test, target_train, target_test)
		percep.append(stats)
		print "MLP"
		stats = multiLayerPerceptron(data_train, data_test, target_train, target_test)
		mlp.append(stats)
		
	
	print "Bayesian Classifier"	
	conf_bayes = np.mean(np.transpose(np.array(bayes_simple)), axis=1)
	base_name = 'naive_bayes'
	plotPREREC(conf_bayes, '../img/'+ base_name + '_rec.png')
	saveConfMat(conf_bayes, base_name + '_conf.csv')
	print conf_bayes

	print "LogisticRegression"
	conf_logreg = np.mean(np.transpose(np.array(logreg)), axis=1)
	base_name = 'log_reg'
	plotPREREC(conf_logreg, '../img/'+ base_name + '_rec.png')
	saveConfMat(conf_logreg, base_name + '_conf.csv')
	print conf_logreg

	print "Quadratic Bayesian Classifier"
	conf_quad_bayes = np.mean(np.transpose(np.array(bayes_quad)), axis=1)
	print conf_quad_bayes
	base_name = 'quad_bayes'
	plotPREREC(conf_quad_bayes, '../img/'+ base_name + '_rec.png')
	saveConfMat(conf_quad_bayes, base_name + '_conf.csv')

	print "Perceptron"
	conf_perc = np.mean(np.transpose(np.array(percep)), axis=1)
	print conf_perc
	base_name = 'perc'
	plotPREREC(conf_perc, '../img/'+ base_name + '_rec.png')
	saveConfMat(conf_perc, base_name + '_conf.csv')

	print "MLP"
	conf_mlp = np.mean(np.transpose(np.array(mlp)), axis=1)
	print conf_mlp
	base_name = 'mlp'
	plotPREREC(conf_mlp, '../img/'+ base_name + '_rec.png')
	saveConfMat(conf_mlp, base_name + '_conf.csv')

if __name__ == "__main__":
    main()
