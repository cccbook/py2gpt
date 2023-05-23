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
    Wb.data = Wb.data - 0.01*Wb.grad # update weights, 相當於 optimizer.step()
    Wb.grad = 0

loss = calculate_loss(train_images, y_train, Wb.data)
print(f'loss in final step {step+1} is {loss}')

'''
$ python mnist.py
loss in step 0 is 17.92461998035962
loss in step 1000 is 0.6774682437270458
loss in step 2000 is 0.5477999727070898
loss in step 3000 is 0.4989018310540714
loss in step 4000 is 0.44975912129728784
loss in step 5000 is 0.4203744240123638
loss in step 6000 is 0.411572470806632
loss in step 7000 is 0.39024896870370307
loss in step 8000 is 0.3769509563066923
loss in step 9000 is 0.3768737910199006
loss in step 10000 is 0.3587365915719602
loss in step 11000 is 0.34737773487143964
loss in step 12000 is 0.37459058392841504
loss in step 13000 is 0.3411812984086028
loss in step 14000 is 0.32203538914621826
loss in step 15000 is 0.32873441998466774
loss in step 16000 is 0.32793335403592927
loss in step 17000 is 0.31231447428300846
loss in step 18000 is 0.3133160779430901
loss in step 19000 is 0.3550552459097412
loss in final step 20000 is 0.31858205117532523
'''