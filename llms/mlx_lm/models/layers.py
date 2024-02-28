from functools import partial

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def rms_norm(x, weight, eps):
    x = x.astype(mx.float32)
    x = x * mx.rsqrt(x.square().mean(-1, keepdims=True) + eps)
    return weight * x.astype(weight.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return rms_norm(x, self.weight, self.eps)


@partial(mx.compile, shapeless=True)
def ln_norm(x, eps, weight=None, bias=None):
    t = x.dtype
    x = x.astype(mx.float32)
    means = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    x = (x - means) * mx.rsqrt(var + eps)
    x = x.astype(t)
    return weight * x + bias if weight is not None else x


class LayerNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        if affine:
            self.bias = mx.zeros((dims,))
            self.weight = mx.ones((dims,))
        self.eps = eps
        self.dims = dims

    def _extra_repr(self):
        return f"{self.dims}, eps={self.eps}, affine={'weight' in self}"

    def __call__(self, x: mx.array) -> mx.array:
        if "weight" in self:
            return ln_norm(x, self.eps, self.weight, self.bias)
        else:
            return ln_norm(x, self.eps)
