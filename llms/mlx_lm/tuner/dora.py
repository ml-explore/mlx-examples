# Copyright © 2024 Apple Inc.

import math

import mlx.core as mx
import mlx.nn as nn


class DoRALinear(nn.Module):
    @staticmethod
    def from_base(
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

    def fuse(self, de_quantize: bool = False):
        linear = self.linear
        bias = "bias" in linear
        weight = self._dequantized_weight()

        # Use the same type as the linear weight
        dtype = weight.dtype

        output_dims, input_dims = weight.shape
        fused_linear = nn.Linear(input_dims, output_dims, bias=False)

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
        Set the self.linear layer and recompute self.m.
        """
        self.linear = linear
        self.m = mx.linalg.norm(self._dequantized_weight().astype(mx.float32), axis=1)

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
        w = self._dequantized_weight()
        y = x @ w.T

        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        out = y + (self.scale * z).astype(x.dtype)

        # Compute the norm of the adapted weights
        adapted = w + (self.scale * self.lora_b.T) @ self.lora_a.T
        denom = mx.stop_gradient(mx.linalg.norm(adapted, axis=1))

        # Remove the norm and scale by the learned magnitude
        out = (self.m / denom).astype(x.dtype) * out

        if "bias" in self.linear:
            out = out + self.linear.bias
        return out


class DoRAEmbedding(nn.Module):
    def from_base(
        embedding: nn.Embedding,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        num_embeddings, dims = embedding.weight.shape

        # TODO support quantized weights in DoRALinear
        if isinstance(embedding, nn.QuantizedLinear):
            raise ValueError("DoRAEmbedding does not yet support quantization.")
        dora_embedding = DoRAEmbedding(
            num_embeddings=num_embeddings,
            dims=dims,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        dora_embedding.set_embedding(embedding)
        return dora_embedding

    def fuse(self, de_quantize: bool = False):
        embedding = self.embedding
        weight = embedding.weight

        # Use the same type as the linear weight if not quantized
        dtype = weight.dtype

        num_embeddings, dims = weight.shape
        fused_embedding = nn.Embedding(num_embeddings, dims)

        lora_a = (self.scale * self.lora_a).astype(dtype)
        lora_b = self.lora_b.astype(dtype)
        weight = weight + lora_a @ lora_b
        norm_scale = self.m / mx.linalg.norm(weight, axis=1)
        fused_embedding.weight = norm_scale[:, None] * weight

        return fused_embedding

    def __init__(
        self,
        num_embeddings: int,
        dims: int,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        super().__init__()

        # Regular embedding layer weights
        self.set_embedding(nn.Embedding(num_embeddings, dims))
        self.dropout = nn.Dropout(p=dropout)

        # Scale for low-rank update
        self.scale = scale

        # Low rank lora weights
        scale = 1 / math.sqrt(num_embeddings)
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_embeddings, r),
        )
        self.lora_b = mx.zeros(shape=(r, dims))

    def set_embedding(self, embedding: nn.Module):
        self.embedding = embedding
        self.m = mx.linalg.norm(embedding.weight, axis=1)

    def __call__(self, x):
        y = self.embedding(x)
        z = self.scale * self.lora_a[x] @ self.lora_b
        out = y + self.dropout(z).astype(y.dtype)

        # Compute the norm of the adapted weights for the individual embeddings
        adapted = y + z
        denom = mx.stop_gradient(mx.linalg.norm(adapted, axis=-1))

        # Remove the norm and scale by the learned magnitude
        out = (self.m[x] / denom)[..., None] * out

        return out

    def as_linear(self, x):
        y = self.embedding.as_linear(x)
        z = (self.dropout(x) @ self.lora_b.T) @ self.lora_a.T
        out = y + (self.scale * z).astype(x.dtype)

        # Compute the norm of the adapted weights
        adapted = self.embedding.weight + (self.scale * self.lora_a) @ self.lora_b
        denom = mx.stop_gradient(mx.linalg.norm(adapted, axis=1))

        # Remove the norm and scale by the learned magnitude
        out = (self.m / denom) * out

        return out
