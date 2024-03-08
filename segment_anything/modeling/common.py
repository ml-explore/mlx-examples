from typing import Type

import mlx.core as mx
import mlx.nn as nn


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def __call__(self, x: mx.array) -> mx.array:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mx.ones(num_channels)
        self.bias = mx.zeros(num_channels)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        u = x.mean(3, keepdims=True)
        s = ((x - u) ** 2).mean(3, keepdims=True)
        x = (x - u) / mx.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x
