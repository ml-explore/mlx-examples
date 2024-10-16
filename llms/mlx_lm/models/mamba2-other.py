# Copyright © 2024 Apple Inc.

import math
from dataclasses import dataclass, field
from typing import Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "mamba2"
    num_heads: int = 128
    head_dim: int = 64
    vocab_size: int = 32768
    hidden_size: int = 4096
    state_size: int = 128
    num_hidden_layers: int = 64
    layer_norm_epsilon: float = 1e-5
    expand: int = 2
    conv_kernel: int = 4
    n_groups: int = 8
    use_bias: bool = False
    use_conv_bias: bool = True
    initializer_range: float = 0.1
    residual_in_fp32: bool = True
    time_step_rank: Union[int, str] = "auto"
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    time_step_limit: Tuple[float, float] = field(default_factory=lambda: (0.0, float("inf")))
    rescale_prenorm_residual: bool = False
    use_cache: bool = True
    rms_norm: bool = True
    chunk_size: int = 256
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if not hasattr(self, "intermediate_size"):
            self.intermediate_size = int(self.expand * self.hidden_size)
        if not hasattr(self, "head_dim"):
            self.head_dim = self.hidden_size // self.num_heads
        if self.time_step_rank == "auto":
            self.time_step_rank = math.ceil(self.hidden_size / 16)


class Mamba2Cache:
    def __init__(self):
        self.cache = [None, None]

    def __setitem__(self, idx, value):
        self.cache[idx] = value

    def __getitem__(self, idx):
        return self.cache[idx]

    @property
    def state(self):
        return self.cache


class MambaRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps

    def __call__(self, hidden_states, gate=None):
        if gate is not None:
            hidden_states = hidden_states * nn.silu(gate)
        variance = mx.mean(hidden_states ** 2, axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states

class DepthWiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, groups=None, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups if groups is not None else in_channels

        # Ensure in_channels and out_channels are the same for depthwise conv
        assert in_channels == out_channels, "In and out channels must be the same for depthwise convolution"
        # Ensure groups is equal to in_channels for depthwise conv
        assert self.groups == in_channels, "Groups must be equal to in_channels for depthwise convolution"

        # Initialize weight with shape (out_channels, kernel_size, 1)
        self.weight = mx.random.normal((out_channels, kernel_size, 1))
        self.bias = mx.zeros((out_channels,)) if bias else None

    def __call__(self, x, cache=None):
        B, L, C = x.shape
        _, K, _ = self.weight.shape

        if cache is not None:
            x = mx.concatenate([cache, x], axis=1)
        else:
            x = mx.pad(x, [(0, 0), (K - 1, 0), (0, 0)])

        y = mx.conv_general(x, self.weight, groups=self.groups)

        if self.bias is not None:
            y = y + self.bias

        return y, x[:, -K + 1 :, :]


class Mamba2Mixer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.intermediate_size = args.intermediate_size
        self.time_step_rank = args.time_step_rank
        self.conv_kernel_size = args.conv_kernel
        self.hidden_size = args.hidden_size
        self.state_size = args.state_size
        self.num_heads = args.num_heads
        self.head_dim = args.hidden_size // args.num_heads
        self.n_groups = args.n_groups

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.state_size
        self.conv1d = DepthWiseConv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=args.use_conv_bias,
            kernel_size=args.conv_kernel,
            groups=self.conv_dim,
            padding=args.conv_kernel - 1
        )

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=args.use_bias
        )

        self.dt_bias = mx.ones((self.num_heads,))
        self.A_log = mx.log(mx.arange(1, self.num_heads + 1))
        self.D = mx.ones((self.num_heads,))

        self.norm = MambaRMSNormGated(self.intermediate_size, eps=args.layer_norm_epsilon)

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=args.use_bias)

    def ssm_step(self, x, state, dt_proj):
        A = -mx.exp(self.A_log)
        D = self.D
        delta = nn.softplus(dt_proj + self.dt_bias)
        
        B, C = mx.split(x, indices_or_sections=[self.state_size * self.n_groups], axis=-1)
        
        B = B.reshape(-1, self.n_groups, self.state_size)
        C = C.reshape(-1, self.n_groups, self.state_size)
        
        if state is None:
            new_state = mx.expand_dims(delta, -1) * B
        else:
            new_state = mx.expand_dims(delta, -1) * (B + state * mx.exp(mx.expand_dims(delta, -1) * A))
        
        y = mx.sum(new_state * C, axis=-1)
        y = y + D * x[:, :self.num_heads]
        return y, new_state

    def __call__(self, x, cache):
        B, T, D = x.shape
        if cache is None:
            cache = [None, None]

        outputs = []
        for t in range(T):
            xt = x[:, t, :]
            xz = self.in_proj(xt)
            
            x_t, z_t, dt_proj = mx.split(
                xz,
                indices_or_sections=[self.conv_dim, self.conv_dim + self.intermediate_size],
                axis=-1
            )

            conv_out, cache[0] = self.conv1d(mx.expand_dims(x_t, 1), cache[0])
            x_t = conv_out.squeeze(1)
            x_t = nn.silu(x_t)
            y_t, cache[1] = self.ssm_step(x_t, cache[1], dt_proj)
            z_t = nn.silu(z_t)
            
            # Print shapes for debugging
            print(f"y_t shape: {y_t.shape}")
            print(f"z_t shape: {z_t.shape}")
            
            # Reshape y_t to (B, num_heads, head_dim)
            y_t_reshaped = y_t.reshape(B, self.num_heads, -1)
            
            # Reshape z_t to (B, num_heads, intermediate_size // num_heads)
            z_t_reshaped = z_t.reshape(B, self.num_heads, -1)
            
            print(f"y_t_reshaped shape: {y_t_reshaped.shape}")
            print(f"z_t_reshaped shape: {z_t_reshaped.shape}")
            
            # Element-wise multiplication (broadcasting across the last dimension)
            output_t = y_t_reshaped * z_t_reshaped
            
            # Reshape to match the expected input of out_proj
            output_t = output_t.reshape(B, -1)
            
            print(f"output_t shape before out_proj: {output_t.shape}")
            print(f"out_proj weight shape: {self.out_proj.weight.shape}")
            
            output_t = self.out_proj(output_t)
            outputs.append(output_t)
        
        output = mx.stack(outputs, axis=1)
        return output


class Mamba2Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mixer = Mamba2Mixer(args)
        self.norm = nn.RMSNorm(args.hidden_size)

    def __call__(self, x: mx.array, cache):
        return self.mixer(self.norm(x), cache) + x


class Mamba2(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Mamba2Block(args) for idx in range(args.num_hidden_layers)]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

    def __call__(
        self,
        inputs: mx.array,
        cache=None
    ):
        hidden_states = self.embeddings(inputs)
        
        if cache is None:
            cache = Mamba2Cache(len(self.layers))

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, cache[i])

        hidden_states = self.norm_f(hidden_states)
        return hidden_states


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.backbone = Mamba2(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        B, T = inputs.shape

        x = self.backbone(inputs, cache)

        if self.args.tie_word_embeddings:
            logits = self.backbone.embeddings.as_linear(x)
        else:
            logits = self.lm_head(x)

        return logits

    def sanitize(self, weights):
        for k, v in weights.items():
            if "conv1d.weight" in k and v.ndim == 3:
                weights[k] = v.moveaxis(2, 1)
        return weights

    def make_cache(self, batch_size: int = 1):
        return [Mamba2Cache() for _ in range(len(self.layers))]

    @property
    def layers(self):
        return self.backbone.layers
