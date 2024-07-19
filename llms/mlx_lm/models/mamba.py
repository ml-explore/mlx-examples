from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import math

import mlx.core as mx
import mlx.nn as nn

from base import BaseModelArgs, KVCache, create_additive_causal_mask


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "mamba"
    dt_rank: Union[int, str] = "auto"
    d_model: int = 12 # hidden_size
    d_inner: int = 2
    vocab_size: int = 623
    n_layer: int = 3# num_hidden_layers
    tie_word_embeddings: bool = False
    use_bias: bool = False
    use_conv_bias: bool = False
    conv_kernel: int = 4
    state_size: int = 16
    expand: int = 2
    time_step_init_scheme: str = "random"
    time_step_max: float = 0.1
    time_step_min: float = 0.001
    time_step_floor: float = 0.0001

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


def clamp(x, min=None, max=None):
    if min is not None:
        mask_lower = x < min
    if max is not None:
        mask_upper = x > max

    if min is not None:
        if max is not None:
            return mx.where(mask_upper, max, mx.where(mask_lower, min, x))
        return mx.where(mask_lower, min, x)

    return mx.where(mask_upper, max, x)

class DepthWiseConv1d(nn.Module):
    def __init__(self, channels, kernel_size, bias, padding):
        super().__init__()


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.in_proj = nn.Linear(args.d_model, 2 * args.d_inner, bias=args.use_bias)
        self.conv1d = DepthWiseConv1d(channels=args.d_inner, kernel_size=args.conv_kernel, bias=args.use_conv_bias, padding=args.conv_kernel-1)
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + 2 * args.state_size, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        dt_init_std = args.dt_rank**-0.5 * args.state_size
        if args.time_step_init_scheme == "constant":
            self.dt_proj.weight = dt_init_std * mx.ones_like(self.dt_proj.weight)
        elif args.time_step_init_scheme == "random":
            self.dt_proj.weight = mx.random.uniform(-dt_init_std, dt_init_std, self.dt_proj.weight.shape)
        else:
            raise NotImplementedError

        dt = clamp(mx.exp(mx.random.uniform(shape=[args.d_inner]) * (math.log(args.time_step_max) - math.log(args.time_step_min)) + math.log(args.time_step_min)), min=args.time_step_floor)
        inv_dt = dt + mx.log1p(-mx.exp(-dt))
        self.dt_proj.bias = inv_dt


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mixer = MambaBlock(args)
        self.norm = nn.RMSNorm(args.d_model)

    def __call__(self, inputs: mx.array, cache=None):
        output, cache = self.mixer(self.norm(inputs), cache)
        output = output + inputs
        return output, cache


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = [ResidualBlock(args) for _ in range(args.n_layer)]
        self.norm_f = nn.RMSNorm(args.d_model)

    def __call__(self, inputs: mx.array, cache=None):
        tokens = self.embedding(inputs)

        for i, layer in enumerate(self.layers):
            h, cache[i] = layer(tokens, cache[i])

        h = self.norm_f(h)
        return h, cache


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.backbone = Mamba(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        out = self.backbone(inputs, cache)

        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        return out


model = Model(ModelArgs())
print(model)
