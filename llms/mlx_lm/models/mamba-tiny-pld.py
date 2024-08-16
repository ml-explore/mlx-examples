from dataclasses import dataclass
from typing import Optional, Union

import math

import torch

# import tokenizer

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    n_layer: int
    use_conv_bias: bool
    expand: int
    pad_vocab_size_multiple: int
    conv_kernel: int
    d_model: int
    state_size: int
    d_inner: int
    initializer_range: float
    use_bias: bool
    time_step_init_scheme: str
    time_step_max: float
    time_step_min: float
    time_step_floor: float
    dt_rank: Union[int, str] = "auto"

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        if self.n_layer is None:
            self.n_layer = self.num_hidden_layers
        if self.d_model is None:
            self.d_model = self.hidden_size
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.in_proj = nn.Linear(args.d_model, 2 * args.d_inner, bias=args.use_bias)
        # self.conv1d = DepthWiseConv1d(channels=args.d_inner, kernel_size=args.conv_kernel, bias=args.use_conv_bias, padding=args.conv_kernel-1)
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.use_conv_bias,
            kernel_size=args.conv_kernel,
            # groups=args.d_inner,
            padding=args.conv_kernel - 1,
        )
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + 2 * args.state_size, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = mx.repeat(mx.arange(1, args.state_size + 1).reshape([1, 16]), repeats=args.d_inner)


        self.A_log = mx.log(A)
        self.D = mx.ones([args.d_inner])

        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.use_bias)

        self.norm = nn.RMSNorm(args.d_model)

    def ssm(self, x):
        d_in, N = self.A_log.shape
        A = -mx.exp(self.A_log.float())
        D = self.D.float()
        delta, B, C = self.x_proj(x).split(split_size=[self.config.dt_rank, N, N], dim=-1)
        delta = nn.softplus(self.dt_proj(delta))
        return self.selective_scan(x, delta, A, B, C, D)

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        deltaA = mx.exp(mx.einsum(delta, A, 'b l d_in, d_in n -> b d_in l n'))
        deltaB_u = mx.einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b d_in l n')
        
        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        x = mx.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            y = mx.einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = mx.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y = y + u * D
    
        return y

    def __call__(self, x):
        _, L, _ = x.shape
        x, r = self.in_proj(x).split([self.args.d_inner, self.args.d_inner], axis=-1)

        x = mx.reshape(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :L]
        x = mx.rearrange(x, 'b d_in l -> b l d_in')
        out = self.ssm(nn.silu(x)) * nn.silu(r)
        return self.out_proj(out) + x

class MambaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = [MambaBlock(args) for _ in range(args.n_layer)]
        self.norm_f = nn.RMSNorm(args.d_model)

    def __call__(self, inputs: mx.array_equal):
        tokens = self.embedding(inputs)
        for i, layer in enumerate(self.layers):
            h = layer(tokens)
        h = self.norm_f(h)
        return h


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = self.backbone = MambaModel(args)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.model.embedding.weight

    def __call__(self, inputs: mx.array):
        h = self.backbone(inputs)
        return self.lm_head(h)

    @property
    def layers(self):
        return self.backbone.layers
    
    # def sanitize(self, weights):
    #     exclude_patterns = [
    #         'backbone.layers.mixer.A_log', 
    #         'backbone.layers.mixer.conv1d.weight',
    #         'backbone.layers.mixer.dt_proj.weight',
    #         'backbone.layers.mixer.in_proj.weight',
    #         'backbone.layers.mixer.dt_proj.bias',
    #         'backbone.layers.mixer.conv1d.bias',
    #         'backbone.layers.mixer.D'
    #     ]
    #     return {
    #         k: v for k, v in weights.items() 
    #         if not any(pattern in k for pattern in exclude_patterns)
    #     }