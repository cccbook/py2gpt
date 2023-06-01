import numpy as np

A = [[[1,2],[3,4]],[[1,2],[3,4]]]
B = A.copy()
M = [[1,2],[3,4]]

AB = np.matmul(A, B)
print('AB=', AB)
AM = np.matmul(A, M)
print('AM=', AM)

V = [1,2]
AV = np.matmul(A, V)
print('AV=', AV)
