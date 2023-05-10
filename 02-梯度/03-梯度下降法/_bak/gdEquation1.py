"""
A X = B ，求 X 是多少？

3x+2y=5
1x+1y=2
"""
import gd2 as gd
import numpy as np

A = np.array([[3.0, 2],[1, 1]])
B = np.array([5.0, 2]) # np.array([[5.0, 2]]).transpose()

def f(p): #  能量函數:計算 ||AX-B||，也就是 ||Y-B||
    X = p.transpose()
    Y = A.dot(X)
    # return np.linalg.norm(Y-B, 1)
    return np.linalg.norm(Y-B, 2)

p = np.array([0.0, 0.0])

gd.gradientDescendent(f, p, step=0.001)
