# Copyright © 2024 Apple Inc.

import math

import mlx.core as mx
import mlx.nn as nn


class DoRALinear(nn.Module):
    @staticmethod
    def from_linear(
        linear: nn.Linear,
        r: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
        scale: float = 10.0,
    ):
        # TODO support quantized weights in DoRALinear
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            raise ValueError("DoRALinear does not yet support quantization.")
        dora_lin = DoRALinear(
            linear=linear,
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            alpha=alpha,
            dropout=dropout,
            scale=scale,
        )

        dora_lin.linear = linear
        return dora_lin

    def to_linear(self, de_quantize: bool = False):
        linear = self.linear
        bias = "bias" in linear
        weight = linear.weight
        m = self.m

        # Use the same type as the linear weight if not quantized
        dtype = weight.dtype

        output_dims, input_dims = weight.shape
        fused_linear = nn.Linear(input_dims, output_dims, bias=bias)

        lora_b = (self.scale * self.lora_b.T).astype(dtype)
        lora_a = self.lora_a.T.astype(dtype)
        fused_linear.weight = weight + lora_b @ lora_a

        fused_norm = mx.linalg.norm(fused_linear.weight, axis=1)

        magnitude = (m / fused_norm)[:, None]

        fused_linear.weight = magnitude * fused_linear.weight

        if bias:
            fused_linear.bias = linear.bias

        return fused_linear

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
        scale: float = 10.0,
    ):
        super().__init__()

        # Regular linear layer weights
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.dropout = nn.Dropout(p=dropout)

        # Scale for low-rank update
        self.scale = scale * (alpha / r)

        # Low rank lora weights
        scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(input_dims, r),
        )
        self.lora_b = mx.zeros(shape=(r, output_dims))
        self.m = mx.linalg.norm(self.linear.weight, axis=1)

    def __call__(self, x):
        dtype = self.linear.weight.dtype
        lb = (self.scale * self.lora_b.T).astype(dtype)
        la = self.lora_a.T.astype(dtype)
        adapted = self.linear.weight + lb.astype(dtype) @ la
        norm = mx.stop_gradient(mx.linalg.norm(adapted, axis=1))
        y = x @ self.linear.weight.T
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        res = y + (self.scale * z).astype(x.dtype)
        if "bias" in self.linear:
            res += self.linear.bias
        return (self.m / norm) * res
