import matplotlib.pyplot as plt
import numpy as np
import gd

# 以下 (x,y) 配對資料，大致在一條 y=a0 x+a1 的線上，我們想用梯度下降法去找出 [a0, a1]
x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
y = np.array([1.9, 3.1, 3.9, 5.0, 6.2], dtype=np.float32)

# 計算 y = a[0]+a[1] x 的函數
def predict(a, xt):
	return a[0]+a[1]*xt

# 最小平方損失函數
def MSE(a, x, y):
	total = 0
	for i in range(len(x)):
		total += (y[i]-predict(a,x[i]))**2
	return total

def loss(p):
	return MSE(p, x, y)

p = [0.0, 0.0]
plearn = gd.gradientDescendent(loss, p, max_loops=1000, dump_period=50)
# Plot the graph
y_predicted = np.array(list(map(lambda t: plearn[0]+plearn[1]*t, x)))
print('y_predicted=', y_predicted)
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, y_predicted, label='Fitted line')
plt.legend()
plt.show()
