# 來源 -- https://github.com/newcodevelop/micrograd/blob/master/mnist.ipynb

from macrograd import Value

from keras.datasets import mnist
import keras
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_images = np.asarray(x_train, dtype=np.float32) / 255.0
test_images = np.asarray(x_test, dtype=np.float32) / 255.0
train_images = train_images.reshape(60000, 784)
test_images = test_images.reshape(10000, 784)
y_train = keras.utils.to_categorical(y_train)

def calculate_loss(X, Y, W):
    return -(1/X.shape[0])*np.sum(np.sum(Y*np.log(np.exp(np.matmul(X, W)) / np.sum(np.exp(np.matmul(X, W)), axis=1)[:, None]), axis=1))

batch_size = 32
steps = 20000
# new initialized weights for gradient descent
Wb = Value(np.random.randn(784, 10))
for step in range(steps):
    ri = np.random.permutation(train_images.shape[0])[:batch_size]
    Xb, yb = Value(train_images[ri]), Value(y_train[ri])
    y_predW = Xb.matmul(Wb)
    probs = y_predW.softmax()
    log_probs = probs.log()

    zb = yb*log_probs

    outb = zb.reduce_sum(axis=1)
    finb = -outb.reduce_sum()  # cross entropy loss
    finb.backward()
    if step % 1000 == 0:
        loss = calculate_loss(train_images, y_train, Wb.data)
        print(f'loss in step {step} is {loss}')
    Wb.data = Wb.data - 0.01*Wb.grad
    Wb.grad = 0

loss = calculate_loss(train_images, y_train, Wb.data)
print(f'loss in final step {step+1} is {loss}')
