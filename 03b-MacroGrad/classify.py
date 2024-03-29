import random
import numpy as np
import matplotlib.pyplot as plt
from macrograd.engine import Tensor
from macrograd.nn import MLP

np.random.seed(1337)
random.seed(1337)

# make up a dataset

from sklearn.datasets import make_moons, make_blobs
X, y = make_moons(n_samples=100, noise=0.1)

y = y*2 - 1 # make y be -1 or 1
'''
# visualize in 2D
plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')

plt.show()
'''
# initialize a model 
# model = MLP(2, [16, 16, 1]) # 2-layer neural network
model = MLP(2, [16, 1]) # 1-layer neural network

# loss function
def loss(batch_size=None):
    
    # inline DataLoader :)
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    # inputs = [list(map(Tensor, xrow)) for xrow in Xb]
    inputs = Tensor(Xb)
    outputs = Tensor(np.transpose([yb]))
    print('outputs.shape=', outputs.shape)
    # forward the model to get scores
    # scores = list(map(model, inputs))
    scores = model(inputs)
    
    # svm "max-margin" loss
    losses = (1. + (-outputs*scores)).relu() # [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
    print('losses.shape=', losses.shape)
    # print('losses.sum()=', losses.sum())
    data_loss = losses.sum() * (1./losses.data.shape[0]) # data_loss = sum(losses) * (1.0 / len(losses))
    # print('data_loss=', data_loss)
    # exit()
    # L2 regularization
    '''
    alpha = 1e-4
    reg_loss = alpha * sum((p*p for p in model.parameters()))
    '''
    total_loss = data_loss # + reg_loss
    
    # also get accuracy
    fscores = scores.data.flatten()
    # print('yb=', yb, 'fscores=', fscores)
    accuracy = [(yi > 0) == (scorei > 0) for yi, scorei in zip(yb, fscores)]
    return total_loss, sum(accuracy) / len(accuracy)

total_loss, acc = loss()
print(total_loss, acc)

# optimization
for k in range(100):
    print('k=', k)
    
    # forward
    total_loss, acc = loss()
    
    # backward
    model.zero_grad()
    total_loss.backward()
    
    # update (sgd)
    learning_rate = 1.0 - 0.9*k/100
    for p in model.parameters():
        print('p.data.shape=', p.data.shape)
        print('p.grad.shape=', p.grad.shape)
        step = learning_rate * p.grad
        print('step.shape=', step.shape)
        p.data -= step
        print('p.data.shape=', p.data.shape)
    
    if k % 1 == 0:
        print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")

# visualize decision boundary

h = 0.25
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Xmesh = np.c_[xx.ravel(), yy.ravel()]
inputs = [list(map(Tensor, xrow)) for xrow in Xmesh]
scores = list(map(model, inputs))
Z = np.array([s.data > 0 for s in scores])
Z = Z.reshape(xx.shape)

fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
