import numgd as ngd
import soft as so
import numpy as np

x = np.array([0.3, 0.5, 0.2])
y = np.array([0.0, 1.0, 0.0])
print('x =', x)
print('y =', y)

s = so.softmax(x)
print('s = softmax(x) =', s)

print('jacobian_softmax(s)=\n', so.jacobian_softmax(s))
print('cross_entropy(y, s)=', so.cross_entropy(y, s))

def num_gradient_cross_entropy(y, s):
    return ngd.grad(lambda s:so.cross_entropy(y, s), s)

print('    gradient_cross_entropy(y, s)=', so.gradient_cross_entropy(y, s))
print('num_gradient_cross_entropy(y, s)=', num_gradient_cross_entropy(y, s))

def loss(y, x):
    s = so.softmax(x)
    return so.cross_entropy(y, s)

def num_error_softmax_input(y, x):
    return ngd.grad(lambda x:loss(y, x), x)

print('    error_softmax_input(y, s)=', so.error_softmax_input(y, s))
print('num_error_softmax_input(y, x)=', num_error_softmax_input(y, x))
