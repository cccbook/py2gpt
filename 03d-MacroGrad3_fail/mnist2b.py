# 來源 -- https://github.com/newcodevelop/micrograd/blob/master/mnist.ipynb

from macrograd.engine import Tensor

from keras.datasets import mnist
import keras
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_images = np.asarray(x_train, dtype=np.float32) / 255.0
test_images = np.asarray(x_test, dtype=np.float32) / 255.0
train_images = train_images.reshape(60000, 784)
test_images = test_images.reshape(10000, 784)
y_train = keras.utils.to_categorical(y_train)

def predict(X,W,W2):
    y_predW = X.matmul(W).relu().matmul(W2)
    print('y_predW=', y_predW)
    return y_predW

def forward(X,Y,W, W2):
    y_predW = predict(X,W,W2)
    probs = y_predW.softmax()
    loss = probs.cross_entropy(Y)
    return loss.sum() # batch sum

batch_size = 32
# steps = 20000
steps = 5000

X = Tensor(train_images); Y = Tensor(y_train) # 全部資料
# new initialized weights for gradient descent
Wb = Tensor(np.random.randn(784, 100))
Wb2 = Tensor(np.random.randn(100, 10))
for step in range(steps):
    ri = np.random.permutation(train_images.shape[0])[:batch_size]
    Xb, yb = Tensor(train_images[ri]), Tensor(y_train[ri]) # Batch 資料
    lossb = forward(Xb, yb, Wb, Wb2)
    lossb.backward()
    if step % 1000 == 0 or step == steps-1:
        loss = forward(X, Y, Wb, Wb2).data/X.data.shape[0]
        print(f'loss in step {step} is {loss}')
    Wb.data = Wb.data - 0.01*Wb.grad # update weights, 相當於 optimizer.step()
    Wb.grad = 0

from sklearn.metrics import accuracy_score
Xb = Tensor(test_images)
ypred = predict(Xb, Wb, Wb2).data
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