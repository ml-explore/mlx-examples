from dataclasses import dataclass
from typing import Optional

import math

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int # d_model
    intermediate_size: int # d_inner
    state_size: int # d_state
    num_hidden_layers: int # n_layer
    layer_norm_epsilon: float
    expand: int
    conv_kernel: int # d_conv
    use_bias: bool # bias
    use_conv_bias: bool # conv_bias
    initializer_range: float
    time_step_rank: int
    time_step_scale: float
    time_step_min: float
    time_step_max: float
    time_step_init_scheme: str
    time_step_floor: float
    rescale_prenorm_residual: bool
    use_cache: bool
    use_mambapy: bool = False # pscan
    dt_rank: str = "auto"

    def __post_init__(self):
        if not hasattr(self, 'hidden_size') and hasattr(self, 'd_model'):
            self.hidden_size = self.d_model
        if not hasattr(self, 'intermediate_size') and hasattr(self, 'd_inner'):
            self.intermediate_size = self.d_inner
        if not hasattr(self, 'state_size') and hasattr(self, 'd_state'):
            self.state_size = self.d_state
        if not hasattr(self, 'num_hidden_layers') and hasattr(self, 'n_layer'):
            self.num_hidden_layers = self.n_layer
        if not hasattr(self, 'conv_kernel') and hasattr(self, 'd_conv'):
            self.conv_kernel = self.d_conv
        if not hasattr(self, 'use_bias') and hasattr(self, 'bias'):
            self.use_bias = self.bias
        if not hasattr(self, 'use_conv_bias') and hasattr(self, 'conv_bias'):
            self.use_conv_bias = self.conv_bias

        self.intermediate_size = self.expand * self.hidden_size
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.hidden_size / 16)


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
        self.weight = mx.random.normal(shape=(channels, 1, kernel_size))
        scale = math.sqrt(1.0 / (channels * kernel_size))
        self.weight *= scale  # Ensure scaling is applied correctly
        if bias:
            self.bias = mx.zeros((channels,))
        else:
            self.bias = None

    def __call__(self, x):
        out = nn.Conv1d(x, self.weight, kernel_size=self.kernel_size, bias=self.bias, padding=self.padding)
        return out
    

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.state_size
        self.conv_kernel_size = args.conv_kernel
        self.intermediate_size = args.intermediate_size
        self.time_step_rank = int(args.time_step_rank)
        self.use_conv_bias = args.use_conv_bias

        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=args.use_bias)

        self.conv1d = DepthWiseConv1d(
            channels=self.intermediate_size,
            kernel_size=self.conv_kernel_size,
            bias=self.use_conv_bias,
            padding=self.conv_kernel_size-1
        )

        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + 2 * self.ssm_state_size, bias=False)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        dt_init_std = args.dt_rank**-0.5 * args.state_size
        if args.time_step_init_scheme == "constant":
            self.dt_proj.weight = dt_init_std * mx.ones_like(self.dt_proj.weight)
        elif args.time_step_init_scheme == "random":
            self.dt_proj.weight = mx.random.uniform(-dt_init_std, dt_init_std, self.dt_proj.weight.shape)
        else:
            raise NotImplementedError

        dt = clamp(mx.exp(mx.random.uniform(shape=[self.intermediate_size]) * (math.log(args.time_step_max) - math.log(args.time_step_min)) + math.log(args.time_step_min)), min=args.time_step_floor)
        self.dt_proj.bias = dt + mx.log1p(-mx.exp(-dt))
        inv_dt = dt + mx.log1p(-mx.exp(-dt))
        self.dt_proj.bias = inv_dt

        A = mx.repeat(mx.arange(1, self.ssm_state_size + 1).reshape([1, self.ssm_state_size]), repeats=self.intermediate_size, axis=0)
        self.A_log = mx.log(A)
        self.D = mx.ones([self.intermediate_size])

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=args.use_bias)

    def ssm(self, x, h):
        A = -mx.exp(self.A_log)
        D = self.D
        delta, B, C = self.x_proj(x).split(split_size=[self.intermediate_size, self.intermediate_size], axis=-1)
        delta = nn.softplus(self.dt_proj(delta))
        deltaA = mx.exp(mx.unsqueeze(delta, -1) * A)
        deltaB = unsqueeze(delta, -1) * unsqueeze(B, 1)
        BX = deltaB * unsqueeze(x, -1)
        if h is None:
            h = mx.zeros([x.shape[0], self.intermediate_size, self.ssm_state_size])
        h = deltaA * h + BX
        y = (h @ mx.unsqueeze(C, -1)).squeeze(2)
        y = y + D * x
        return y, h

    def __call__(self, x, cache = None):
        h, inputs = cache
        x, z = self.in_proj(x).split(indices_or_sections=2, axis=1)
        x_cache = unsqueeze(x, 1)
        x = self.conv1d(mx.concatenate([inputs, x_cache], axis=1))[:, self.conv_kernel_size-1, :] # (B, ED)
        y, h = self.ssm(nn.silu(x), h)
        output = y * nn.silu(z)
        inputs = mx.concatenate([inputs[:, 1:, :], x_cache], axis=1)
        cache.update(h, inputs)
        return self.out_proj(output), cache


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mixer = MambaBlock(args)
        self.norm = nn.RMSNorm(args.hidden_size)

    def __call__(self, inputs: mx.array, cache):
        output, cache = self.mixer(self.norm(inputs), cache)
        output = output + inputs
        return output, cache


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [ResidualBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm_f = nn.RMSNorm(args.hidden_size)

    def __call__(self, inputs: mx.array, cache=None):
        tokens = self.embeddings(inputs)
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
        self.model_type = args.model_type
        self.backbone = Mamba(args)
        # self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        out, cache = self.backbone(inputs, cache)
        out = self.backbone.embeddings.as_linear(out)
        return out, cache

    @property
    def layers(self):
        return self.backbone.layers
    
    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_hidden_layers

    @property
    def n_kv_heads(self):
        return self.args.num_hidden_layers
    
    def make_cache(self):
        # return [(None, mx.zeros([1, self.args.conv_kernel-1, self.args.intermediate_size])) for _ in range(self.args.num_hidden_layers)]
        return [(None, mx.zeros([1, self.backbone.layers[0].mixer.conv_kernel_size-1, self.backbone.layers[0].mixer.intermediate_size])) for _ in range(len(self.backbone.layers))]
    
    def generate(self, input_ids: mx.array, n_tokens_to_gen: int = 50, sample: bool = True, temperature: float = 1.0, top_k: int = None):
        self.eval()

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        caches = self.make_cache()

        for i in range(input_ids.shape[1] + n_tokens_to_gen - 1):
            next_token_logits, caches = self(input_ids[:, i], caches)

            if i+1 >= input_ids.shape[1]:

                if top_k is not None:
                    values = mx.topk(next_token_logits, k=top_k) # (1, k) ordered from lowest to biggest
                    mask = next_token_logits < (values[:, 0, None])
                    next_token_logits = mx.where(mask, -5000, next_token_logits) # TODO -mx.inf is problematic for now

                if sample and temperature > 0:
                    next_token = mx.random.categorical(next_token_logits * (1/temperature), num_samples=1)
                else:
                    next_token = mx.argmax(next_token_logits, axis=-1)[:, None]

                input_ids = mx.concatenate([input_ids, next_token], axis=1)

        self.train()
        return input_ids
