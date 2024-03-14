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
    """
    Layer normalization for input tensor x.

    Args:
        x (np.ndarray): Input tensor.
        eps (float, optional): Small value to avoid division by zero.
        weight (np.ndarray, optional): Weight tensor for normalization.
        bias (np.ndarray, optional): Bias tensor for normalization.

    Returns:
        np.ndarray: Normalized tensor.
    """
    t = x.dtype
    x = x.astype(mx.float32)

    # Compute mean and variance along the last dimension
    means = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)

    # Normalize the input tensor
    x = (x - means) * mx.rsqrt(var + eps)
    x = x.astype(t)

    # Apply weight and bias if provided
    if weight is not None:
        x = x * weight
    if bias is not None:
        x = x + bias
    return x


class LayerNorm(nn.Module):
    def __init__(
        self, dims: int, eps: float = 1e-5, affine: bool = True, bias: bool = True
    ):
        super().__init__()
        self.eps = eps
        self.dims = dims
        self.affine = affine

        if affine:
            self.weight = mx.ones((dims,))
            self.bias = mx.zeros((dims,)) if bias else None

    def _extra_repr(self):
        return f"{self.dims}, eps={self.eps}, affine={'weight' in self}"

    def __call__(self, x: mx.array) -> mx.array:
        if self.affine:
            if self.bias is not None:
                return ln_norm(x, self.eps, self.weight, self.bias)
            else:
                return ln_norm(x, self.eps, self.weight)
        else:
            return ln_norm(x, self.eps)
