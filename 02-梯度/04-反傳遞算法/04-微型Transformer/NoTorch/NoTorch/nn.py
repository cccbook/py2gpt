from typing import Callable, List
from NoTorch.tensor import Tensor
import math
import numpy as np


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)


class Dense(Module):
    """
    Fully Connected Layer, with customizable activation function
    """

    def __init__(
        self,
        in_neurons: int,
        out_neurons: int,
        batch_size: int = 1,
        use_bias: bool = True,
        activation: Callable = lambda x: x,
    ):
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.use_bias = use_bias
        self.activation = activation
        self.weights = Tensor(
            np.array(
                [
                    np.longdouble(np.random.randn(in_neurons))
                    * math.sqrt(2.0 / in_neurons)
                    for _ in range(out_neurons)
                ]
            )
        )
        self.bias = Tensor(np.ones(shape=[out_neurons]).astype(np.longdouble) * 0.01)
        self.batch_size = batch_size

    def __call__(self, x: np.ndarray):
        return self.activation(Tensor.mat_vec_mul(self.weights, x) + self.bias)

    def parameters(self):
        return [self.weights, self.bias]

    def __repr__(self):
        return f"Dense layer with in: {self.in_neurons} out: {self.out_neurons}"


class MLP(Module):
    """
    A basic MLP, with a variable number of hidden layers and sizes, relu activation by default
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_sizes: List[int],
        hidden_activation: Callable = lambda x: x.relu(),
    ):
        f = hidden_activation

        if len(hidden_sizes) == 0:
            self.layers = [Dense(in_features, out_features)]

        elif len(hidden_sizes) == 1:
            self.layers = [
                Dense(in_features, hidden_sizes[0], activation=f),
                Dense(hidden_sizes[0], out_features),
            ]

        else:
            self.layers = [Dense(in_features, hidden_sizes[0], activation=f)]
            self.layers += [
                Dense(hidden_sizes[i], hidden_sizes[i + 1], activation=f)
                for i in range(len(hidden_sizes) - 1)
            ]
            self.layers += [Dense(hidden_sizes[-1], out_features)]

    def __call__(self, x: np.ndarray):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params
