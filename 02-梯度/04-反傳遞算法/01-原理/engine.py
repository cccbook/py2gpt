
class Value: # 含有梯度的值節點
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data # 目前值
        self.grad = 0 # 梯度預設為 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None # 反傳遞函數
        self._prev = set(_children) # 前面的網路節點
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other): # 加法的正向傳遞
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward(): # 加法的反向傳遞
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other): # 乘法的正向傳遞
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward(): # 乘法的反向傳遞
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other): # 次方的正向傳遞
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward(): # 次方的反向傳遞
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self): # relu 的正向傳遞
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward(): # relu 的反向傳遞
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v): # 建立網路拓譜結構
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo): # 反向排列
            v._backward()
    # 以下這些運算，由於 + * 已被 override ，所以反向傳遞會自動建構，不需再自己加入反向傳遞函數
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
        return f"Value(data={self.data}, grad={self.grad})"
