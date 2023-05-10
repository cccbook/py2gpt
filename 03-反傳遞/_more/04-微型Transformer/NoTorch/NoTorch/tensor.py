from __future__ import annotations
from typing import Union, List
import numpy as np
import itertools


class Tensor:
    """
    Matrix with differentiable operations

    supports: +, -, *, /, **, log, exp, relu, sigmoid

    operations are supported with np.ndarray, int, float, list of int, list of float, and other Tensors

    Refactored from https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
    """

    unique_id = itertools.count()

    def __init__(
        self,
        data: Union[np.ndarray, int, float, list],
        _children: tuple = (),
        _op: str = "None",
    ):
        self.data = Tensor._validate_init_input(data)
        self.id = next(self.unique_id)
        self._children = _children
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op

    def __add__(self, other):
        """
        Addition operation
        """

        other: Tensor = Tensor._validate_input(other)

        out = Tensor(np.add(self.data, other.data), (self, other), _op="+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        """
        Multiplication operation
        """

        other: Tensor = Tensor._validate_input(other)

        out = Tensor(np.multiply(self.data, other.data), (self, other), _op="*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        """
        Exponentiation operation
        """
        other: Tensor = Tensor._validate_input(other)

        out = Tensor(
            self.data**other.data, (self, other), _op="pow"
        )

        def _backward():
            self.grad += (other.data * self.data ** (other.data - 1)) * out.grad
            other.grad += (self.data ** other.data) * np.log(self.data)

        out._backward = _backward

        return out

    def __rpow__(self, other):
        other: Tensor = Tensor._validate_input(other)

        return other**self

    def log(self):
        out = Tensor(np.log(self.data), (self,), _op="log")

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        """
        Rectified Non-linearity
        """
        out = Tensor(self.data * (self.data > 0), (self,), _op="relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        """
        e^x operation
        """
        out = Tensor(np.exp(self.data), (self,), _op="exp")

        def _backward():
            self.grad += np.exp(self.data)
        
        out._backward = _backward

        return out

    def __gt__(self, other):
        return self.data > Tensor._validate_input(other).data

    def __lt__(self, other):
        return self.data < Tensor._validate_input(other).data

    def __ge__(self, other):
        return self.data >= Tensor._validate_input(other).data

    def __eq__(self, other):
        return self.data == Tensor._validate_input(other).data

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        nodes = []
        visited = set()

        def topological_sort(v):
            if v not in visited:
                visited.add(v)
                for child in set(v._children):
                    topological_sort(child)
                nodes.append(v)

        topological_sort(self)

        self.grad = np.ones_like(self.grad)

        for v in reversed(nodes):
            v._backward()

    def __repr__(self):
        return f"Tensor with val {self.data} and grad {self.grad}"

    def __hash__(self):
        return self.id

    def __getitem__(self, key):
        raise NotImplementedError("Tensor indexing not implemented")

    @staticmethod
    def dot(a, b):
        """
        Vector dot product
        """
        a = Tensor._validate_input(a)
        b = Tensor._validate_input(b)

        return Tensor.sum1d(a*b)

    @staticmethod
    def mat_vec_mul(mat, vec):
        """
        Multiply matrix with vector using np.matmul
        """
        mat = Tensor._validate_input(mat)
        vec = Tensor._validate_input(vec)

        out = Tensor(np.matmul(mat.data, vec.data), (mat, vec), _op="mat_vec_mul")

        def _backward():
            mat.grad += np.outer(out.grad.T, vec.data)
            vec.grad += np.matmul(mat.data.T, out.grad)

        out._backward = _backward
        return out

    @staticmethod
    def sum1d(tensor_in):
        """
        Sum across all values in a 1d tensor in 1 operation,
        mitigating graph blowup
        """
        assert len(tensor_in.data.shape) == 1, "Input tensor must be 1d"

        out = Tensor(np.longdouble([np.sum(tensor_in.data)]), (tensor_in,), _op="sum1d")

        def _backward():
            tensor_in.grad += np.full_like(tensor_in.grad, out.grad)

        out._backward = _backward

        return out

    @staticmethod
    def cat1d(tensors: List[Tensor]):
        """
        Concatenate a list of 1 dimensional Tensors along first axis
        """
        out = Tensor(
            np.concatenate([t.data for t in tensors], axis=0),
            tuple(tensors),
            _op="cat1d",
        )

        def _backward():
            for i in range(len(tensors)):
                tensors[i].grad += out.grad[i]

        out._backward = _backward

        return out

    @staticmethod
    def _validate_init_input(input):
        assert isinstance(input, (np.ndarray, int, float, list)), f"{type(input)}"

        if isinstance(input, int):
            return np.longdouble([float(input)])

        elif isinstance(input, float):
            return np.longdouble([input])

        elif isinstance(input, list):
            return np.longdouble([float(d) for d in input])

        elif isinstance(input, np.ndarray):
            assert input.dtype in (
                np.longdouble,
                np.half,
                np.float16,
                np.float32,
                np.float64,
                np.float128,
                np.single,
                np.double,
            ), "dtype must be float"
            return input

    @staticmethod
    def _validate_input(input):

        if isinstance(input, np.ndarray):
            return Tensor(input, ())

        elif isinstance(input, Tensor):
            return input

        elif isinstance(input, (int, float, list)):
            return Tensor(input, ())

        else:
            raise NotImplementedError(
                f"Tensor operations for {type(input)} not implemented"
            )
