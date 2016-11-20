import numpy as np
import csv
import matplotlib.pyplot as plt

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

plt.imshow(cm,interpolation='nearest')
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

