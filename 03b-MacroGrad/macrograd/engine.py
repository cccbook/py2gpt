# 來源 -- https://github.com/newcodevelop/micrograd/blob/master/micrograd/engine.py
import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data)
        self.grad = np.zeros(self.data.shape)
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    @property
    def shape(self):
        return self.data.shape
    
    def __add__(self, other):
        # assert self.shape == other.shape
        other = other if isinstance(other, Tensor) else Tensor(np.zeros(self.shape)+other) # 讓維度一致
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            # print('self.grad = ', self.grad)
            # print('other.grad = ', other.grad)
            # print('out.grad = ', out.grad, 'op=', out._op)
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.zeros(self.shape)+other) # 讓維度一致
        # other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            print('self.shape=', self.shape)
            print('other.shape=', other.shape)
            print('out.shape=', out.shape)
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
                        
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'relu') # Tensor(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def matmul(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.matmul(self.data , other.data), (self, other), 'matmul')

        def _backward():
            self.grad += np.dot(out.grad,other.data.T)
            other.grad += np.dot(self.data.T,out.grad)            
            
        out._backward = _backward

        return out

    def softmax(self):
        out =  Tensor(np.exp(self.data) / np.sum(np.exp(self.data), axis=1)[:, None], (self,), 'softmax')
        softmax = out.data

        def _backward():
            s = np.sum(out.grad * softmax, 1)
            t = np.reshape(s, [-1, 1]) # reshape 為 n*1
            self.grad += (out.grad - t) * softmax

        out._backward = _backward

        return out

    def log(self):
        out = Tensor(np.log(self.data),(self,),'log')

        def _backward():
            self.grad += out.grad/self.data
        out._backward = _backward

        return out    
    
    def sum(self,axis = None):
        out = Tensor(np.sum(self.data,axis = axis), (self,), 'SUM')
        
        def _backward():
            output_shape = np.array(self.data.shape)
            output_shape[axis] = 1
            tile_scaling = self.data.shape // output_shape
            grad = np.reshape(out.grad, output_shape)
            self.grad += np.tile(grad, tile_scaling)
            
        out._backward = _backward

        return out

    def cross_entropy(self, yb):
        log_probs = self.log()
        zb = yb*log_probs
        outb = zb.sum(axis=1)
        loss = -outb.sum()  # cross entropy loss
        return loss # 本函數不用指定反傳遞，因為所有運算都已經有反傳遞了，所以直接呼叫 loss.backward() 就能反傳遞了

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            #print(v)
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
