import math
from typing import Any

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
        # TODO remove when input_dims and output_dims are attributes
        # on linear and quantized linear
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        dora_lin = DoRALinear(
            linear=linear,
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            alpha=alpha,
            dropout=dropout,
            scale=scale,
        )
        #dora_lin.linear = linear
        return dora_lin
    
    
    def to_linear(self, de_quantize: bool = False):
        linear = self.linear
        bias = "bias" in linear
        weight = linear.weight
        m=self.m

        is_quantized = isinstance(linear, nn.QuantizedLinear)

        # Use the same type as the linear weight if not quantized
        dtype = weight.dtype

        if is_quantized:
            dtype = mx.float16
            weight = mx.dequantize(
                weight,
                linear.scales,
                linear.biases,
                linear.group_size,
                linear.bits,
            )
        output_dims, input_dims = weight.shape
        fused_linear = nn.Linear(input_dims, output_dims, bias=bias)

        lora_b = (self.scale * self.lora_b.T).astype(dtype)
        lora_a = self.lora_a.T.astype(dtype)
        fused_linear.weight = weight + lora_b @ lora_a

        fused_norm=fused_linear.weight.square().sum(axis=0,keepdims=True).sqrt()

        fused_linear.weight=fused_linear.weight/fused_norm*m

        if bias:
            fused_linear.bias = linear.bias
        if is_quantized and not de_quantize:
            fused_linear = nn.QuantizedLinear.from_linear(
                fused_linear,
                linear.group_size,
                linear.bits,
            )

        return fused_linear
    
    
    
    def __init__(
            self,
            linear: nn.Module,
            input_dims:int,
            output_dims:int,
            r: int =8,
            alpha:float=16,
            dropout:float =0.0,
            scale:float =10.0,
    ):
        super().__init__()

        # Regular linear layer weights
        self.linear = linear
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
        self.m = mx.array(self.linear.weight.square().sum(axis=0,keepdims=True).sqrt())
    def __call__(self,x) :
        dtype=self.linear.weight.dtype
        lb = (self.scale * self.lora_b.T).astype(dtype)
        la = self.lora_a.T.astype(dtype)
        adapted=self.linear.weight+lb.astype(dtype) @ la
        norm= adapted.square().sum(axis=0,keepdims=True).sqrt()
        y = self.linear(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        res=y+(self.scale*z).astype(x.dtype)
        return res/norm*self.m


