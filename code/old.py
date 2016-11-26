import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.naive_bayes import GaussianNB

variable_names = ['Diag', 'Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean', 'Smoothness Mean', 'Compactness Mean', 'Concavity Mean', 'Concave Points Mean',    'Symmetry Mean', 'Fractal Dimension Mean',	'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE', 'Smoothness SE', 'Compactness SE', 'Concavity SE',     'Concave points SE', 'Symmetry SE', 'Fractal Dimension SE',	'Radius Worst',	'Texture Worst', 'Perimeter Worst', 'Area Worst', 'Smoothness Worst',     'Compactness Worst',	'Concavity Worst', 'Concave Points Worst',	'Symmetry Worst', 'Fractal Dimension Worst']

# Loads csv discarting the first line (variable names) 
def loadCsv(filename):
	b = []
	with open(filename, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		i = 0
		for row in reader:
			if i == 0: 
				i += 1
				continue
			b.append([float(j) for j in row])
	return b

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
	matrix=np.asmatrix(nparray)
	output=matrix[:,0]
	output_np=np.asarray(output)

	# Calculate zscore without output var
	c = np.delete(nparray,0,1)
	ctrans = np.transpose(c)
	ctranspad = stats.zscore(ctrans)
	cpad = np.transpose(ctranspad)

	# Add output var again
	data_zscore = np.concatenate((output_np, cpad), axis=1)
	return data_zscore
	
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
def removeOutliersByDistance(data, dist, pout):
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
	return data_clean


## Distance matrix and outline removal
dataset = loadCsv('data2.csv')
dataset_np = np.array(dataset)
#sort by class 1st column
dataset_np = dataset_np[dataset_np[:,0].argsort()]
dataset_zs = zscore(dataset_np)
dist = distanceMatrix(np.delete(dataset_zs,0,1))
dataset_clean = removeOutliersByDistance(dataset_zs, dist, 0.1)
print dataset_clean
dist_clean = distanceMatrix(np.delete(dataset_clean,0,1))

## 
gnb = GaussianNB()
data = np.delete(dataset_clean,0,1)
matrix=np.asmatrix(dataset_clean)
output=matrix[:,0]
target=np.asarray(output) 
y_pred = gnb.fit(data, target.ravel()).predict(data)

mislabeled = 0
for i in range(0,np.shape(data)[0]):
	if target[i] != y_pred[i]:
		mislabeled = mislabeled + 1
print "Number of mislabeled points out of a total", np.shape(data)[0],  "points :", mislabeled 
