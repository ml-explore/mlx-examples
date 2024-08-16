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
    hidden_size: int
    intermediate_size: int
    state_size: int
    num_hidden_layers: int
    layer_norm_epsilon: float
    expand: int
    conv_kernel: int
    use_bias: bool
    use_conv_bias: bool
    initializer_range: float
    time_step_rank: int
    time_step_scale: float
    time_step_min: float
    time_step_max: float
    time_step_init_scheme: str
    time_step_floor: float
    rescale_prenorm_residual: bool
    use_cache: bool
    use_mambapy: bool = False
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
        if not hasattr(self, 'num_hidden_layers') and hasattr(self, 'n_layers'):
            self.num_hidden_layers = self.n_layers
        if not hasattr(self, 'conv_kernel') and hasattr(self, 'd_conv'):
            self.conv_kernel = self.d_conv
        if not hasattr(self, 'use_bias') and hasattr(self, 'bias'):
            self.use_bias = self.bias
        if not hasattr(self, 'use_conv_bias') and hasattr(self, 'conv_bias'):
            self.use_conv_bias = self.conv_bias

        self.intermediate_size = self.expand * self.hidden_size
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.hidden_size / 16)

class DepthWiseConv1d(nn.Module):
    def __init__(self, channels, kernel_size, bias, padding):
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.bias = bias
        self.padding = padding
        self.weight = mx.random.normal(shape=(self.channels, 1, self.kernel_size))
        scale = math.sqrt(1.0 / (self.channels * self.kernel_size))
        self.weight *= scale  # Ensure scaling is applied correctly
        if bias:
            self.bias = mx.zeros((self.channels,))
        else:
            self.bias = None

    def __call__(self, x):
        B, D, L = x.shape
        assert D == self.channels, f"Input channels ({D}) must match the initialized channels ({self.channels})."
        print("FORWARD PASS THROUGH CONV")
        print(self.kernel_size)
        print(self.weight)
        out = nn.Conv1d(x, self.weight, kernel_size=self.kernel_size, bias=self.bias, padding=self.padding)
        return out


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

        dt = clamp(mx.exp(
            mx.random.uniform(shape=[args.intermediate_size]) * (math.log(args.time_step_max) - math.log(args.time_step_min)) + math.log(args.time_step_min)
        ), min=args.time_step_floor)
        inv_dt = dt + mx.log1p(-mx.exp(-dt))
        self.dt_proj.bias = inv_dt

        A = mx.repeat(mx.arange(1., self.ssm_state_size + 1.).reshape([1, self.ssm_state_size]), repeats=self.intermediate_size, axis=0)
        self.A_log = mx.log(A)
        self.D = mx.ones([self.intermediate_size])

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=args.use_bias)

    def ssm(self, x, h):
        A = -mx.exp(self.A_log)  # (ED, N)
        D = self.D

        deltaBC = self.x_proj(x)  # (B, dt_rank+2*N)

        delta, B, C = mx.split(deltaBC, indices_or_sections=[self.time_step_rank, self.time_step_rank+self.ssm_state_size], axis=-1)  # (B, dt_rank), (B, N), (B, N)
        delta = nn.softplus(self.dt_proj(delta))  # (B, ED)

        deltaA = mx.exp(mx.expand_dims(delta, -1) * A)  # (B, ED, N)
        deltaB = mx.expand_dims(delta, -1) * mx.expand_dims(B, 1)  # (B, ED, N)

        BX = deltaB * mx.expand_dims(x, -1)  # (B, ED, N)

        if h is None:
            h = mx.zeros([x.shape[0], self.intermediate_size, self.ssm_state_size])  # (B, ED, N)

        h = deltaA * h + BX  # (B, ED, N)

        y = (h @ mx.expand_dims(C, -1)).squeeze(2)  # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x
        return y, h
    
    def __call__(self, x, cache):
        h, inputs = cache
        x, z = self.in_proj(x).split(indices_or_sections=2, axis=-1)  # (B, ED), (B, ED)
        # x branch
        x_cache = mx.expand_dims(x, 1)  # (B, 1, ED)
        # Ensure inputs has the correct shape
        if inputs.ndim == 2:
            inputs = mx.expand_dims(inputs, 1)  # Add a dimension if it's missing

        print(f"inputs shape: {inputs.shape}")
        print(f"x_cache shape: {x_cache.shape}")
            
        conv_input = mx.concatenate([inputs, x_cache], axis=1)  # (B, d_conv, ED) <---------- Here is the problem ValueError: [concatenate] All the input arrays must have the same number of dimensions. However, got arrays with dimensions 3 and 4. ||| inputs shape: (1, 3, 1536) x_cache shape: (1, 1, 5, 1536)
        x = self.conv1d(conv_input)[:, -1, :]  # (B, ED)
        y, h = self.ssm(nn.silu(x), h)
        output = y * nn.silu(z) # * z branch
        # prepare cache for next call
        inputs = mx.concatenate([inputs[:, 1:, :], x_cache], axis=1)  # (B, d_conv-1, ED)
        return self.out_proj(output), (h, inputs) # (B, D), cache

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

    def __call__(self, inputs: mx.array, cache):
        tokens = self.embeddings(inputs)
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
            input_ids = mx.expand_dims(input_ids, 0)

        caches = self.make_cache()

        for i in range(input_ids.shape[1] + n_tokens_to_gen - 1):
            next_token_logits, caches = self(input_ids[:, i], caches)

            if i+1 >= input_ids.shape[1]:

                if top_k is not None:
                    values = mx.topk(next_token_logits, k=top_k)
                    mask = next_token_logits < (values[:, 0, None])
                    next_token_logits = mx.where(mask, -5000, next_token_logits)

                if sample and temperature > 0:
                    next_token = mx.random.categorical(next_token_logits * (1/temperature), num_samples=1)
                else:
                    next_token = mx.argmax(next_token_logits, axis=-1)[:, None]

                input_ids = mx.concatenate([input_ids, next_token], axis=1)

        self.train()
        return input_ids
    
    def generate_step(self, input_ids: mx.array, sample: bool = True, temperature: float = 1.0, top_k: int = None):
        self.eval()

        if input_ids.ndim == 1:
            input_ids = mx.expand_dims(input_ids, 0)

        caches = self.make_cache()

        # Generate the next token logits
        next_token_logits, caches = self(input_ids, caches)

        # Apply top_k filtering if specified
        if top_k is not None:
            values = mx.topk(next_token_logits, k=top_k) # (1, k) ordered from lowest to highest
            mask = next_token_logits < (values[:, -1, None])
            next_token_logits = mx.where(mask, -5000, next_token_logits) # -mx.inf is problematic for now

        # Sample the next token or take the argmax based on the temperature
        if sample and temperature > 0:
            next_token = mx.random.categorical(next_token_logits * (1 / temperature), num_samples=1)
        else:
            next_token = mx.argmax(next_token_logits, axis=-1)[:, None]

        # Concatenate the next token to the input_ids
        input_ids = mx.concatenate([input_ids, next_token], axis=1)

        self.train()
        return input_ids, caches