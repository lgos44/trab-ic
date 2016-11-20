import numpy as np
import csv
import matplotlib.pyplot as plt



titles = ['Diag', 'Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean', 'Smoothness Mean', 'Compactness Mean', 'Concavity Mean', 'Concave Points Mean',    'Symmetry Mean', 'Fractal Dimension Mean',	'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE', 'Smoothness SE', 'Compactness SE', 'Concavity SE',     'Concave points SE', 'Symmetry SE', 'Fractal Dimension SE',	'Radius Worst',	'Texture Worst', 'Perimeter Worst', 'Area Worst', 'Smoothness Worst',     'Compactness Worst',	'Concavity Worst', 'Concave Points Worst',	'Symmetry Worst', 'Fractal Dimension Worst']

b = []
with open('data2.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='|')
	i = 0
	for row in reader:
		if i == 0: 
			i += 1
			continue
		b.append([float(j) for j in row])

c = np.array(b)
d = np.transpose(c)
cm = np.corrcoef(d)


temp = []
for i in range(0, 30):
	for j in range(0, 30):
		if cm[i][j] > 0.9 and i!=j:
			if (j,i) not in temp:
				temp.append((i,j))
				print '\item ' + titles[i]+', ' + titles[j] + ' ' + str(cm[i][j])

v = np.linspace(-1.0, 1.0, 20, endpoint=True)
plt.imshow(cm,interpolation='nearest', vmin=-1, vmax=1)
plt.colorbar()
plt.show()

dim = np.shape(c)
print (dim)
dist = np.zeros((dim[0],dim[0]))
for i in range(0, dim[0]):
	for j in range(0, dim[0]):
		dist[i][j] = np.linalg.norm(c[i]-c[j])
print(dist)
plt.imshow(dist,interpolation='nearest')
plt.colorbar()
plt.show()

