import numpy as np
A = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
B=np.asmatrix(A)
C=B[:,0]
D=np.asarray(C)
E = np.delete(A,0,1)
print C
print D
print A
print np.concatenate((D, E), axis=1)
print np.concatenate((C, C), axis=1)


