# Copyright © 2024 Apple Inc.

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

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
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    expand: int = 2
    conv_kernel: int = 4
    n_groups: int = 8
    use_bias: bool = False
    use_conv_bias: bool = True
    hidden_act: str = "silu"
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
    def __init__(self, num_layers):
        self.cache = [[None, None] for _ in range(num_layers)]

    def __getitem__(self, idx):
        return self.cache[idx]

    def __setitem__(self, idx, value):
        self.cache[idx] = value


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


class Mamba2Mixer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.intermediate_size = args.intermediate_size
        self.conv_kernel_size = args.conv_kernel
        self.state_size = args.state_size
        self.num_heads = args.num_heads
        self.head_dim = args.head_dim
        self.n_groups = args.n_groups
        self.time_step_rank = args.time_step_rank

        projection_size = self.intermediate_size + self.intermediate_size + 2 * self.n_groups * self.state_size + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=args.use_bias
        )

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=args.use_conv_bias,
            kernel_size=args.conv_kernel,
            groups=self.conv_dim,
            padding=args.conv_kernel - 1,
        )

        self.act = nn.SiLU()
        self.dt_bias = mx.ones((self.num_heads,))
        self.A_log = mx.log(mx.arange(1, self.num_heads + 1, dtype=mx.float32))
        self.D = mx.ones((self.num_heads,))

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=args.use_bias)
        self.norm = MambaRMSNormGated(self.intermediate_size, eps=args.layer_norm_epsilon)

    def ssm_step(self, x, dt, state):
        A = -mx.exp(self.A_log)
        D = self.D

        deltaBC = self.in_proj(x)
        gate, conv_state, time_step = mx.split(
            deltaBC,
            [self.intermediate_size, self.intermediate_size + 2 * self.n_groups * self.state_size],
            axis=-1
        )

        conv_state = conv_state.transpose(0, 2, 1)
        conv_out = self.conv1d(conv_state)
        conv_out = conv_out.transpose(0, 2, 1)
        conv_out = self.act(conv_out)

        x_and_conv_out, B, C = mx.split(
            conv_out,
            [self.intermediate_size, self.n_groups * self.state_size],
            axis=-1
        )

        dt = nn.softplus(time_step + self.dt_bias)
        dt = mx.clip(dt, self.args.time_step_min, self.args.time_step_max)

        B = B.reshape(-1, self.num_heads, self.head_dim, self.state_size)
        C = C.reshape(-1, self.num_heads, self.head_dim, self.state_size)

        dA = mx.exp(dt[:, :, None, None] * A[None, :, None, None])
        dB = dt[:, :, None, None] * B

        new_state = state * dA + x_and_conv_out[:, :, None, None] * dB
        y = mx.sum(new_state * C, axis=-1)
        y = y + D[None, :, None] * x_and_conv_out

        y = self.norm(y.reshape(-1, self.intermediate_size), gate)
        output = self.out_proj(y)

        return output, new_state

    def __call__(
        self,
        x: mx.array,
        cache = None
    ):
        B, L, _ = x.shape

        if cache[0] is not None:  # Using cached state
            conv_state, ssm_state = cache
            x = x[:, -1:]
            output, new_ssm_state = self.ssm_step(x, None, ssm_state)
            cache[1] = new_ssm_state  # Update SSM state in cache
        else:
            conv_state, ssm_state = None, None
            outputs = []
            for t in range(L):
                x = x[:, t:t+1]
                output, ssm_state = self.ssm_step(x, None, ssm_state)
                outputs.append(output)
            output = mx.concatenate(outputs, axis=1)
            cache[1] = ssm_state  # Store final SSM state in cache

        # Update conv state in cache
        new_conv_state = x[:, -self.conv_kernel_size:]
        cache[0] = new_conv_state
        
        return output


class Mamba2Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.residual_in_fp32 = args.residual_in_fp32
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.mixer = Mamba2Mixer(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.mixer(self.norm(inputs), cache_params=cache)
        r = inputs + h
        return r


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

    def __call__(
        self,
        inputs: mx.array,
        cache=None
    ):
        B, T = inputs.shape

        x = self.backbone(inputs, cache)

        if self.args.tie_word_embeddings:
            logits = self.backbone.embeddings.as_linear(x)
        else:
            logits = self.lm_head(x)
        return logits
    
    def sanitize_mabey(self, weights):
        for k, v in weights.items():
            if "conv1d.weight" in k and v.ndim == 3:
                weights[k] = v.moveaxis(2, 1)
        return weights
    
    def make_cache(self, batch_size: int = 1):
        return Mamba2Cache(len(self.backbone.layers))
    
    @property
    def layers(self):
        return self.backbone.layers
