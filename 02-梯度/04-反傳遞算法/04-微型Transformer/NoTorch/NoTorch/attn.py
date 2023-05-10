"""
Attention for Transformers
"""

from typing import List, Union
from NoTorch.nn import Module, MLP
from NoTorch.tensor import Tensor
import numpy as np
import math


class MultiHeadAttention(Module):
    """
    Multi-Head Scaled Dot Product Attention
    """

    def __init__(self, input_dim: int, heads: int):
        def _w_init() -> Tensor:
            return Tensor(
                np.array(
                    [
                        np.longdouble(np.random.randn(input_dim))
                        * math.sqrt(2.0 / input_dim)
                        for _ in range(input_dim)
                    ]
                )
            )

        self.w_query = [_w_init() for _ in range(heads)]
        self.w_key = [_w_init() for _ in range(heads)]
        self.w_value = [_w_init() for _ in range(heads)]

        self.heads = heads
        self.input_dim = input_dim

    def __call__(self, x: List[Union[Tensor, np.ndarray]]) -> List[Tensor]:
        
        def y_i_r(i: int, r: int) -> Tensor:
            """
            Get vector output for token i, for head r
            """

            query_key_dot = [
                (
                    Tensor.mat_vec_mul(self.w_query[r], x[i])
                    * Tensor.mat_vec_mul(self.w_key[r], x[j])
                    / math.sqrt(len(x))
                ).exp()
                for j in range(len(x))
            ]
            qk_sum = sum(query_key_dot)
            return sum([qkd / qk_sum for qkd in query_key_dot]) * Tensor.mat_vec_mul(
                self.w_value[r], x[i]
            )

        return [Tensor.cat1d([y_i_r(i, r) for r in range(self.heads)]) for i in range(len(x))]

    def parameters(self):
        return self.w_query + self.w_key + self.w_value


class TransformerLayer(Module):
    """
    Linear layer stacked on Multihead attention
    """

    def __init__(self, input_dim: int, output_dim: int, heads: int):
        self.input_dim = input_dim
        self.heads = heads
        self.attn = MultiHeadAttention(input_dim, heads)
        self.linear = MLP(input_dim * heads, output_dim, hidden_sizes=[16, 16])

    def __call__(self, x: List[Tensor]) -> List[Tensor]:
        return [self.linear(token) for token in self.attn(x)]

    def parameters(self):
        return self.attn.parameters() + self.linear.parameters()