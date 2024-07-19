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
    n_layer: int = 3 # num_hidden_layers
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
    pscan: bool = False

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

def unsqueeze(x, axis):
    assert axis <= len(x.shape)
    if axis >= 0:
        new_shape = x.shape[:axis] + tuple([1]) + x.shape[axis:]
    else:
        new_shape = x.shape + tuple([1])
    return x.reshape(new_shape)


class DepthWiseConv1d(nn.Module):
    def __init__(self, channels, kernel_size, bias, padding):
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = padding

        self.conv1d = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, bias=True, padding=padding)

        indices = mx.arange(channels)
        mask = mx.zeros_like(self.conv1d.weight)
        mask[indices, :, indices] = 1
        self.conv1d.weight *= mask

    def __call__(self, x):
        return self.conv1d(x)


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
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
        self.dt_proj.bias = dt + mx.log1p(-mx.exp(-dt))

        A = mx.repeat(mx.arange(1., 16 + 1.).reshape([1, 16]), repeats=args.d_inner, axis=0)
        self.A_log = mx.log(A)
        self.D = mx.ones([args.d_inner])

        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.use_bias)


    def ssm_step(self, x, h):
        A = -mx.exp(self.A_log)
        D = self.D
        deltaBC = self.x_proj(x)
        delta, B, C = mx.split(deltaBC, indices_or_sections=[self.args.dt_rank, self.args.dt_rank+self.args.state_size], axis=-1)
        delta = nn.softplus(self.dt_proj(delta))
        deltaA = mx.exp(unsqueeze(delta, -1) * A)
        deltaB = unsqueeze(delta, -1) * unsqueeze(B, 1)
        BX = deltaB * unsqueeze(x, -1)
        if h is None:
            h = mx.zeros([x.shape[0], self.args.d_inner, self.args.d_state])
        h = deltaA * h + BX
        y = (h @ unsqueeze(C, -1)).squeeze(2)
        y = y + D * x
        return y, h

    def ssm(self, x):
        A = -mx.exp(self.A_log) # (ED, N)
        D = self.D

        deltaBC = self.x_proj(x) # (B, L, dt_rank+2*N)

        delta, B, C = mx.split(deltaBC, indices_or_sections=[self.args.dt_rank, self.args.dt_rank+self.args.state_size], axis=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta = nn.softplus(self.dt_proj(delta)) # (B, L, ED)
        if self.args.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)
        return y


    def selective_scan(self, x, delta, A, B, C, D):
        deltaA = mx.exp(unsqueeze(delta, -1) * A) # (B, L, ED, N)
        deltaB = unsqueeze(delta, -1) * unsqueeze(B, 2) # (B, L, ED, N)

        BX = deltaB * unsqueeze(x, -1) # (B, L, ED, N)

        hs = pscan(deltaA, BX)

        y = (hs @ unsqueeze(C, -1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        _, L, _ = x.shape

        deltaA = mx.exp(unsqueeze(delta, -1) * A) # (B, L, ED, N)
        deltaB = unsqueeze(delta, -1) * unsqueeze(B, 2) # (B, L, ED, N)

        BX = deltaB * unsqueeze(x, -1) # (B, L, ED, N)

        h = mx.zeros([x.shape[0], self.args.d_inner, self.args.state_size]) # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = mx.stack(hs, axis=1)

        y = (hs @ unsqueeze(C, -1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y


    def __call__(self, inputs: mx.array, cache = None):
        _, L, _ = inputs.shape

        if cache is not None:
            h, inputs = cache

        x, z = self.in_proj(inputs).split(indices_or_sections=2, axis=2)
        x_cache = unsqueeze(x, 1)
        # x = self.conv1d(mx.concatenate([inputs, x_cache], axis=1))[:, self.args.conv_kernel-1, :]
        x = self.conv1d(x)[:, :L, :]
        # y, h = self.ssm_step(nn.silu(x), h)
        output = self.ssm(nn.silu(x)) * nn.silu(z)
        # inputs = mx.concatenate([inputs[:, 1:, :], x_cache], axis=1)
        return self.out_proj(output), None # (h, inputs)


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mixer = MambaBlock(args)
        self.norm = nn.RMSNorm(args.d_model)

    def __call__(self, inputs: mx.array, cache: Optional[mx.array] = None):
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

        if cache is None:
            cache = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            h, cache[i] = layer(tokens, cache[i])

        h = self.norm_f(h)
        return h, cache


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
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

logits = model(mx.array([[3, 3, 3]]))
print(logits)
