import numpy as np
import csv
import matplotlib.pyplot as plt

b = []

titles = ['Diag', 'Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean', 'Smoothness Mean', 'Compactness Mean', 'Concavity Mean', 'Concave Points Mean',    'Symmetry Mean', 'Fractal Dimension Mean',	'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE', 'Smoothness SE', 'Compactness SE', 'Concavity SE',     'Concave points SE', 'Symmetry SE', 'Fractal Dimension SE',	'Radius Worst',	'Texture Worst', 'Perimeter Worst', 'Area Worst', 'Smoothness Worst',     'Compactness Worst',	'Concavity Worst', 'Concave Points Worst',	'Symmetry Worst', 'Fractal Dimension Worst']


with open('data2.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='|')
	i = 0
	for row in reader:
		if i == 0: 
			i += 1
			continue
		b.append([float(j) for j in row])

c = np.array(b)
c = c[c[:,1].argsort()]
dim = np.shape(c)

## Calculating distance matrix
dist = np.zeros((dim[0],dim[0]))
for i in range(0, dim[0]):
	for j in range(0, dim[0]):
		dist[i][j] = np.linalg.norm(c[i]-c[j])

plt.imshow(dist,interpolation='nearest', origin='lower')
plt.colorbar()
plt.show()

## Distance mean
mi = np.mean(dist, axis=1)
mi_tuple = []
index = 0
for value in mi:
	mi_tuple.append((index,value))
	index += 1

## Sorting by distance 
mi_tuple.sort(key=lambda tup: tup[1])
pout = 0.1
numreg = len(mi_tuple)*(1-pout)

outliers = []
for index in range(int(numreg),len(mi_tuple)):
	outliers.append(mi_tuple[index][0])

## Removing outliers
print c 
cclean = c
cclean = np.delete(c, (outliers), axis=0)

## Dist matrix after cleaning
dim = np.shape(cclean)
dist_clean = np.zeros((dim[0],dim[0]))
for i in range(0, dim[0]):
	for j in range(0, dim[0]):
		dist_clean[i][j] = np.linalg.norm(cclean[i]-cclean[j])

plt.imshow(dist_clean,interpolation='nearest',origin='lower')
plt.colorbar()
plt.show()




