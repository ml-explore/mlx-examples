# Copyright © 2024 Apple Inc.

import math
from dataclasses import dataclass, field
from typing import Tuple, Union, Optional

import mlx.nn as nn
import mlx.core as mx

from .base import BaseModelArgs
from .cache import Mamba2Cache

# python -m mlx_lm.generate --model rokyang/mamba2-130m-hf  --prompt "hello how are you."

@dataclass
class ModelArgs(BaseModelArgs):
    num_heads: int
    head_dim: int
    vocab_size: int
    hidden_size: int
    state_size: int
    num_hidden_layers: int
    layer_norm_epsilon: float
    expand: int
    conv_kernel: int
    n_groups: int
    use_bias: bool
    use_conv_bias: bool
    initializer_range: float 
    residual_in_fp32: bool
    time_step_min: float
    time_step_max: float
    time_step_floor: float
    rescale_prenorm_residual: bool
    use_cache: bool
    rms_norm: bool
    chunk_size: int
    tie_word_embeddings: bool
    time_step_limit: Tuple[float, float] = field(default_factory=lambda: (0.0, float("inf")))
    time_step_rank: Union[int, str] = "auto"
    model_type: str = "mamba2"

    def __post_init__(self):
        if not hasattr(self, "intermediate_size"):
            self.intermediate_size = int(self.expand * self.hidden_size)
        if not hasattr(self, "head_dim"):
            self.head_dim = self.hidden_size // self.num_heads
        if self.time_step_rank == "auto":
            self.time_step_rank = math.ceil(self.hidden_size / 16)

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
    def __init__(self, args, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = args.hidden_size
        self.intermediate_size = args.intermediate_size
        self.num_heads = args.num_heads
        self.head_dim = args.head_dim
        self.ssm_state_size = args.state_size
        self.n_groups = args.n_groups
        self.conv_kernel_size = args.conv_kernel
        self.use_conv_bias = args.use_conv_bias
        self.use_bias = args.use_bias
        self.time_step_min = args.time_step_min
        self.time_step_max = args.time_step_max
        self.chunk_size = args.chunk_size
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=args.use_bias
        )
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            self.conv_kernel_size,
            groups=self.conv_dim,
            bias=self.use_conv_bias
        )
        self.act = nn.SiLU()
        self.norm = MambaRMSNormGated(self.intermediate_size, eps=args.layer_norm_epsilon)
        self.out_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=self.use_bias
        )

        self.A_log = mx.zeros(self.num_heads)
        self.D = mx.ones(self.num_heads)
        self.dt_bias = mx.zeros(self.num_heads)
    
    def __call__(self, input_states, cache):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        projected_states = self.in_proj(input_states)
        
        # Calculate the sizes of each split
        total_size = projected_states.shape[-1]
        remaining_size = total_size - self.intermediate_size - self.conv_dim - self.num_heads
        d_mlp = remaining_size // 2
        sizes = [
            d_mlp,
            d_mlp,
            self.intermediate_size,
            self.conv_dim,
            self.num_heads
        ]
        
        # Perform the split operation
        split_result = mx.split(projected_states, sizes, axis=-1)
        
        # Print debug information
        print(f"Number of split parts: {len(split_result)}")
        print(f"Shapes of split parts: {[part.shape for part in split_result]}")
        
        # Flexibly handle the split result
        _, _, _, gate, hidden_states, dt = split_result

        if cache is not None:
            conv_state = cache.conv_states[self.layer_idx]
            if conv_state is None:
                # Initialize conv_state if it's None
                conv_state = mx.zeros((batch_size, 1, self.conv_kernel_size, hidden_states.shape[-1]))
            
            conv_state = mx.roll(conv_state, -1, -2)  # Roll along the kernel dimension
            
            # Reshape hidden_states to match conv_state dimensions
            hidden_states_reshaped = hidden_states[:, None, None, :]
            
            conv_state = mx.concat([conv_state[:, :, :-1, :], hidden_states_reshaped], axis=-2)
            cache.conv_states[self.layer_idx] = conv_state
            
            # Adjust the convolution operation
            hidden_states = mx.sum(conv_state * self.conv1d.weight[:, :, None, :], axis=(-2, -1))
            
            if self.use_conv_bias:
                hidden_states += self.conv1d.bias
            hidden_states = self.act(hidden_states)[:, None, :]
        else:
            hidden_states = hidden_states.transpose(0, 2, 1)
            hidden_states = self.act(self.conv1d(hidden_states)).transpose(0, 2, 1)

        hidden_states, B, C = mx.split(hidden_states, [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size], axis=-1)

        A = -mx.exp(self.A_log.astype(mx.float32))
        dt = nn.softplus(dt + self.dt_bias)
        dt = mx.clip(dt, self.time_step_min, self.time_step_max)

        hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).astype(mx.float32)
        B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).astype(mx.float32)
        C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).astype(mx.float32)

        B = mx.repeat(B, repeats=self.num_heads // self.n_groups, axis=2)
        C = mx.repeat(C, repeats=self.num_heads // self.n_groups, axis=2)

        if cache is not None and cache.seqlen_offset > 0:
            ssm_state = cache.ssm_states[self.layer_idx]
            dA = mx.exp(dt[:, None, :, None] * A[None, :, None, None])
            dB = dt[:, None, :, None] * B
            dBx = dB * hidden_states[:, :, :, None]
            ssm_state = ssm_state * dA + dBx
            cache.ssm_states[self.layer_idx] = ssm_state

            y = mx.sum(ssm_state * C[:, None, :, :], axis=-1)
            D = self.D[None, :, None].expand(self.D.shape[0], self.head_dim)
            y = y + hidden_states * D

            y = y.reshape(batch_size, -1)[:, None, :]
        else:
            # Implement chunked computation here (simplified version)
            pad_size = self.chunk_size - (seq_len % self.chunk_size)
            hidden_states_padded = mx.pad(hidden_states, [(0, 0), (0, pad_size), (0, 0), (0, 0)])
            B_padded = mx.pad(B, [(0, 0), (0, pad_size), (0, 0), (0, 0)])
            C_padded = mx.pad(C, [(0, 0), (0, pad_size), (0, 0), (0, 0)])

            chunks = seq_len // self.chunk_size + (1 if pad_size > 0 else 0)
            y_list = []
            ssm_state = mx.zeros((batch_size, self.num_heads, self.head_dim, self.ssm_state_size))

            for i in range(chunks):
                chunk_start = i * self.chunk_size
                chunk_end = (i + 1) * self.chunk_size
                chunk_h = hidden_states_padded[:, chunk_start:chunk_end]
                chunk_B = B_padded[:, chunk_start:chunk_end]
                chunk_C = C_padded[:, chunk_start:chunk_end]

                chunk_dt = dt[:, chunk_start:chunk_end]
                dA = mx.exp(chunk_dt[:, :, None, None] * A[None, None, :, None])
                dB = chunk_dt[:, :, None, None] * chunk_B
                dBx = dB * chunk_h[:, :, :, None]

                chunk_y = mx.zeros_like(chunk_h)
                for j in range(self.chunk_size):
                    ssm_state = ssm_state * dA[:, j] + dBx[:, j]
                    chunk_y[:, j] = mx.sum(ssm_state * chunk_C[:, j], axis=-1)

                y_list.append(chunk_y)

            y = mx.concat(y_list, axis=1)
            if pad_size > 0:
                y = y[:, :seq_len]

            D = self.D[None, :, None].expand(self.D.shape[0], self.head_dim)
            y = y + hidden_states * D
            y = y.reshape(batch_size, seq_len, -1)

        y = self.norm(y, gate)
        contextualized_states = self.out_proj(y.astype(dtype))

        return contextualized_states


class Mamba2Block(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.mixer = Mamba2Mixer(args, layer_idx)
        self.norm = nn.RMSNorm(args.hidden_size)

    def __call__(self, x: mx.array, cache):
        return self.mixer(self.norm(x), cache) + x


class Mamba2(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Mamba2Block(args, idx) for idx in range(args.num_hidden_layers)]
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

        print(logits)
        print(logits.shape)

        return logits

    def sanitize(self, weights):
        for k, v in weights.items():
            if "conv1d.weight" in k and v.ndim == 3:
                weights[k] = v.moveaxis(2, 1)
        return weights

    def make_cache(self):
        return [Mamba2Cache(self.args.num_hidden_layers) for _ in range(len(self.layers))]

    @property
    def layers(self):
        return self.backbone.layers
