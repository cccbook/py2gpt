# https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py

  # ***** mlops (unary) *****

  def logical_not(self): return mlops.Eq.apply(*self._broadcasted(False))
  def neg(self): return mlops.Neg.apply(self) if self.dtype != dtypes.bool else self.logical_not()
  def contiguous(self): return mlops.Contiguous.apply(self)
  def contiguous_backward(self): return mlops.ContiguousBackward.apply(self)
  def log(self): return mlops.Log.apply(self.cast(least_upper_float(self.dtype)))
  def log2(self): return self.log()/math.log(2)
  def exp(self): return mlops.Exp.apply(self.cast(least_upper_float(self.dtype)))
  def exp2(self): return mlops.Exp.apply(self*math.log(2))
  def relu(self): return mlops.Relu.apply(self)
  def sigmoid(self): return mlops.Sigmoid.apply(self.cast(least_upper_float(self.dtype)))
  def sin(self): return mlops.Sin.apply(self.cast(least_upper_float(self.dtype)))
  def sqrt(self): return mlops.Sqrt.apply(self.cast(least_upper_float(self.dtype)))
  def rsqrt(self): return self.reciprocal().sqrt()
  def cos(self): return ((math.pi/2)-self).sin()
  def tan(self): return self.sin() / self.cos()

  # ***** math functions (unary) *****

  def trunc(self: Tensor) -> Tensor: return self.cast(dtypes.int32).cast(self.dtype)
  def ceil(self: Tensor) -> Tensor: return (self > (b := self.trunc())).where(b+1, b)
  def floor(self: Tensor) -> Tensor: return (self < (b := self.trunc())).where(b-1, b)
  def round(self: Tensor) -> Tensor:
    return ((self > 0) == ((b := self.cast(dtypes.int32) / 2.0).cast(dtypes.int32) == b)).where((self - 0.5).ceil(), (self + 0.5).floor())

  def square(self): return self*self
  def clip(self, min_, max_): return self.maximum(min_).minimum(max_)
  def abs(self): return self.relu() + (-self).relu()
  def sign(self): return ((self.float()) / (self.float().abs() + 1e-12)).cast(self.dtype)
  def reciprocal(self): return mlops.Reciprocal.apply(self.cast(least_upper_float(self.dtype)))

  # ***** activation functions (unary) *****

  def elu(self, alpha=1.0): return self.relu() - alpha*(1-self.exp()).relu()
  def celu(self, alpha=1.0): return self.maximum(0) + (alpha * ((self / alpha).exp() - 1)).minimum(0)
  def swish(self): return self * self.sigmoid()
  def silu(self): return self.swish()   # The SiLU function is also known as the swish function.
  def relu6(self): return self.relu() - (self-6).relu()
  def hardswish(self): return self * (self+3).relu6() * (1/6)
  def tanh(self): return 2.0 * ((2.0 * self).sigmoid()) - 1.0
  def sinh(self): return (self.exp() - self.neg().exp()) / 2
  def cosh(self): return (self.exp() + self.neg().exp()) / 2
  def atanh(self): return ((1 + self)/(1 - self)).log() / 2
  def asinh(self): return (self + (self.square() + 1).sqrt()).log()
  def acosh(self): return (self + (self.square() - 1).sqrt()).log()
  def hardtanh(self, min_val=-1, max_val=1): return self