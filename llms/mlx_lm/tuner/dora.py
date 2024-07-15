# Copyright © 2024 Apple Inc.

import math

import mlx.core as mx
import mlx.nn as nn


class DoRALinear(nn.Module):
    @staticmethod
    def from_linear(
        linear: nn.Linear,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        # TODO remove when input_dims and output_dims are attributes
        # on linear and quantized linear
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        dora_lin = DoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        dora_lin.set_linear(linear)
        return dora_lin

    def to_linear(self, de_quantize: bool = False):
        linear = self.linear
        bias = "bias" in linear
        weight = self._dequantized_weight()

        # Use the same type as the linear weight
        dtype = weight.dtype

        output_dims, input_dims = weight.shape
        fused_linear = nn.Linear(input_dims, output_dims, bias=bias)

        lora_b = (self.scale * self.lora_b.T).astype(dtype)
        lora_a = self.lora_a.T.astype(dtype)
        weight = weight + lora_b @ lora_a
        norm_scale = self.m / mx.linalg.norm(weight, axis=1)
        fused_linear.weight = norm_scale[:, None] * weight

        if bias:
            fused_linear.bias = linear.bias

        if self._is_quantized() and not de_quantize:
            fused_linear = nn.QuantizedLinear.from_linear(
                fused_linear,
                linear.group_size,
                linear.bits,
            )
        return fused_linear

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
        bias: bool = False,
    ):
        super().__init__()

        # Regular linear layer weights
        self.set_linear(nn.Linear(input_dims, output_dims, bias=bias))
        self.dropout = nn.Dropout(p=dropout)

        # Scale for low-rank update
        self.scale = scale

        # Low rank lora weights
        scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(input_dims, r),
        )
        self.lora_b = mx.zeros(shape=(r, output_dims))

    def set_linear(self, linear):
        """
        Set the self.linear layer and recompute self.m with respect to quantization
        """
        self.linear = linear
        # self.m needs to be recomputed when self.linear changes
        # which doesn't happen in from_linear when dora_lin.linear = linear
        self.m = mx.linalg.norm(self._dequantized_weight(), axis=1).astype(mx.float32)

    def _dequantized_weight(self):
        """
        Return the weight of linear layer and dequantize it if is quantized
        """
        weight = self.linear.weight
        if self._is_quantized():
            weight = mx.dequantize(
                weight,
                self.linear.scales,
                self.linear.biases,
                self.linear.group_size,
                self.linear.bits,
            )
        return weight

    def _is_quantized(self):
        return isinstance(self.linear, nn.QuantizedLinear)

    def __call__(self, x):
        # Regular LoRA (without a bias)
        if self._is_quantized():
            # Use quantized_matmul instead of dequantizing for efficiency
            y = mx.quantized_matmul(
                x,
                self.linear.weight,
                scales=self.linear.scales,
                biases=self.linear.biases,
                transpose=True,
                group_size=self.linear.group_size,
                bits=self.linear.bits,
            )
        else:
            y = x @ self.linear.weight.T

        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        out = y + (self.scale * z).astype(x.dtype)

        # Compute the norm of the adapted weights
        adapted = (
            self._dequantized_weight() + (self.scale * self.lora_b.T) @ self.lora_a.T
        )
        denom = mx.stop_gradient(mx.linalg.norm(adapted, axis=1))

        # Remove the norm and scale by the learned magnitude
        out = (self.m / denom) * out

        if "bias" in self.linear:
            out = out + self.linear.bias
        return out
