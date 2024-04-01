import random
from macrograd.engine import Tensor
import numpy as np

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def parameters(self):
        return []
    
class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Tensor(random.uniform(-1.0,1.0)) for _ in range(nin)]
        self.b = Tensor(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b) # sum(iterable, start)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

class Linear(Module):

    def __init__(self, nin, nout, **kwargs):
        self.W = Tensor(np.random.rand(nin, nout))
        self.b = Tensor(np.random.rand(nout))

    def __call__(self, x):
        out = x.matmul(self.W) + self.b
        return out

    def parameters(self):
        return [self.W, self.b]


class ReLU(Module):

    def __call__(self, x):
        return x.relu()

    def __repr__(self):
        return f"ReLU"
