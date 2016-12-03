#!/usr/bin/python

import numpy as np
import matplotlib.mlab as mlab
import csv

import matplotlib.pyplot as plt



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





def histogram():
	np.set_printoptions(suppress=True)
	data, target = loadCsv('data2.csv')
	
	for j in range(0,30):
		data_class0=[]
		data_class1=[]
	
		i=0
	
	
		for row in target:
			if row==1:
				data_class0.append(data[i][j])
		
			if row==-1:
				data_class1.append(data[i][j])
			
			i+=1
	
		mu0 = np.mean(data_class0)
		sigma0 = np.std(data_class0)
		mu1 = np.mean(data_class1)
		sigma1 = np.std(data_class1)
		label_class=['benigno','maligno']
	
		n, bins, patches = plt.hist([data_class0, data_class1], bins='auto', normed=1, label =label_class)  

		y0 = mlab.normpdf(bins, mu0, sigma0)
		y1= mlab.normpdf(bins, mu1, sigma1)

	
		plt.plot(bins, y0, 'r--')
		plt.plot(bins, y1, 'r--')
	
		plt.legend(prop={'size': 10})
		plt.title(variable_names[j+1])
		plt.show()


def main():

	histogram()
	
	
	
if __name__ == "__main__":
    main()
