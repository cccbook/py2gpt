from macrograd.engine import Tensor
import numpy as np

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Linear(Module):

    def __init__(self, nin, nout, **kwargs):
        self.W = Tensor(np.random.rand(nin, nout))
        self.b = Tensor(np.random.rand(nout))

    def __call__(self, x):
        # print('x.shape=', x.data.shape)
        # print('W.shape=', self.W.data.shape)
        out = x.matmul(self.W) + self.b
        return out

    def parameters(self):
        return [self.W, self.b]

class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Linear(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

