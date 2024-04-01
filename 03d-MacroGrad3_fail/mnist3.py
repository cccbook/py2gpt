# 來源 -- https://github.com/newcodevelop/micrograd/blob/master/mnist.ipynb

from macrograd.engine import Tensor
from macrograd.nn import Linear, ReLU, Net

from keras.datasets import mnist
import keras
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_images = np.asarray(x_train, dtype=np.float32) / 255.0
test_images = np.asarray(x_test, dtype=np.float32) / 255.0
train_images = train_images.reshape(60000, 784)
test_images = test_images.reshape(10000, 784)
y_train = keras.utils.to_categorical(y_train)

# net = Net([Linear(784, 10), ReLU(), Linear(10, 10)])
# net = Net([Linear(784, 100), ReLU(), Linear(100, 10)])
net = Net([Linear(784, 10)])

def predict(X):
    y_predW = net(X)
    return y_predW

def forward(X,Y):
    # print('X=', X)
    y_predW = predict(X)
    # print('y_predW=', y_predW)
    probs = y_predW.softmax()
    #print('probs.shape=', probs.shape)
    #print('probs=', probs)
    loss = probs.cross_entropy(Y)
    #print('loss.shape=', loss.shape)
    #print('loss=', loss)
    return loss.sum() # batch sum

batch_size = 32
# steps = 20000
steps = 5000

X = Tensor(train_images); Y = Tensor(y_train) # 全部資料
# new initialized weights for gradient descent
for step in range(steps):
    ri = np.random.permutation(train_images.shape[0])[:batch_size]
    Xb, yb = Tensor(train_images[ri]), Tensor(y_train[ri]) # Batch 資料
    lossb = forward(Xb, yb)
    # print('lossb=', lossb)
    lossb.backward()
    if step % 1000 == 0 or step == steps-1:
        loss = forward(X, Y).data/X.data.shape[0]
        print(f'loss in step {step} is {loss}')
    for w in net.parameters():
        w.data = w.data - 0.01*w.grad # update weights, 相當於 optimizer.step()
        w.zero_grad()

from sklearn.metrics import accuracy_score
Xb = Tensor(test_images)
ypred = predict(Xb).data
print(f'accuracy on test data is {accuracy_score(np.argmax(ypred,axis = 1),y_test)*100} %')

'''
$ python mnist2.py
loss in step 0 is 110.73387244771637
loss in step 1000 is 0.8965394767411253
loss in step 2000 is 0.6085738263522903
loss in step 3000 is 0.5067586502031575
loss in step 4000 is 0.44567157408167823
loss in step 4999 is 0.40398629173244466
accuracy on test data is 89.53999999999999 %
'''