# Copyright © 2024-2025 Apple Inc.

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from .cache import MambaCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    state_size: int
    num_hidden_layers: int
    conv_kernel: int
    use_bias: bool
    use_conv_bias: bool
    time_step_rank: int
    tie_word_embeddings: bool = True
    use_bcdt_rms: bool = False
    mixer_rms_eps: float = 1e-6

    def __post_init__(self):
        if not hasattr(self, "hidden_size") and hasattr(self, "d_model"):
            self.hidden_size = self.d_model
        if not hasattr(self, "intermediate_size") and hasattr(self, "d_inner"):
            self.intermediate_size = self.d_inner
        if not hasattr(self, "state_size") and hasattr(self, "d_state"):
            self.state_size = self.d_state
        if not hasattr(self, "num_hidden_layers") and hasattr(self, "n_layer"):
            self.num_hidden_layers = self.n_layer
        if not hasattr(self, "num_hidden_layers") and hasattr(self, "n_layers"):
            self.num_hidden_layers = self.n_layers
        if not hasattr(self, "conv_kernel") and hasattr(self, "d_conv"):
            self.conv_kernel = self.d_conv
        if not hasattr(self, "use_bias") and hasattr(self, "bias"):
            self.use_bias = self.bias
        if not hasattr(self, "use_conv_bias") and hasattr(self, "conv_bias"):
            self.use_conv_bias = self.conv_bias

        if self.time_step_rank == "auto":
            self.time_step_rank = math.ceil(self.hidden_size / 16)
        if self.model_type == "falcon_mamba":
            self.use_bcdt_rms = True


class DepthWiseConv1d(nn.Module):
    def __init__(self, channels, kernel_size, bias=True, padding=0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight = mx.random.normal((self.channels, kernel_size, 1))
        self.bias = mx.zeros((channels,)) if bias else None

    def __call__(self, x, cache=None):
        B, L, C = x.shape
        groups, K, _ = self.weight.shape

        if cache is not None:
            x = mx.concatenate([cache, x], axis=1)
        else:
            x = mx.pad(x, [(0, 0), (K - 1, 0), (0, 0)])

        y = mx.conv_general(x, self.weight, groups=groups)

        if self.bias is not None:
            y = y + self.bias

        return y, x[:, -K + 1 :, :]


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
        self.use_bcdt_rms = args.use_bcdt_rms
        if self.use_bcdt_rms:
            self.mixer_norm = lambda x: mx.fast.rms_norm(
                x, mx.ones(x.shape[-1], x.dtype), eps=args.mixer_rms_eps
            )

        self.in_proj = nn.Linear(
            self.hidden_size, self.intermediate_size * 2, bias=args.use_bias
        )

        self.conv1d = DepthWiseConv1d(
            channels=self.intermediate_size,
            kernel_size=self.conv_kernel_size,
            bias=self.use_conv_bias,
            padding=self.conv_kernel_size - 1,
        )

        self.x_proj = nn.Linear(
            self.intermediate_size,
            self.time_step_rank + 2 * self.ssm_state_size,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        A = mx.repeat(
            mx.arange(1.0, self.ssm_state_size + 1.0).reshape([1, self.ssm_state_size]),
            repeats=self.intermediate_size,
            axis=0,
        )
        self.A_log = mx.log(A)
        self.D = mx.ones([self.intermediate_size])

        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=args.use_bias
        )

    def ssm_step(self, x, A, state=None):
        D = self.D
        deltaBC = self.x_proj(x)
        delta, B, C = map(
            self.mixer_norm if self.use_bcdt_rms else lambda x: x,
            mx.split(
                deltaBC,
                [self.time_step_rank, self.time_step_rank + self.ssm_state_size],
                axis=-1,
            ),
        )
        if self.use_bcdt_rms:
            delta, B, C = map(self.mixer_norm, (delta, B, C))
        delta = nn.softplus(self.dt_proj(delta))
        new_state = mx.expand_dims(delta * x, -1) * mx.expand_dims(B, 1)
        if state is not None:
            new_state += state * mx.exp(mx.expand_dims(delta, -1) * A)
        y = (new_state @ mx.expand_dims(C, -1)).squeeze(2)
        y = y + D * x
        return y, new_state

    def _process_sequence(self, x, conv_cache, state_cache):
        B, T, D = x.shape
        xz = self.in_proj(x)
        x, z = xz.split(indices_or_sections=2, axis=-1)

        conv_out, new_conv_cache = self.conv1d(x, conv_cache)
        x = nn.silu(conv_out)

        A = -mx.exp(self.A_log)

        outputs = []
        current_state = state_cache
        y = []
        for t in range(T):
            y_t, current_state = self.ssm_step(x[:, t], A, current_state)
            y.append(y_t)
        y = mx.stack(y, axis=1)
        z = self.out_proj(nn.silu(z) * y)
        return z, (new_conv_cache, current_state)

    def __call__(self, x, cache):
        if cache is None:
            conv_cache, state_cache = None, None
        else:
            conv_cache, state_cache = cache[0], cache[1]

        output, (new_conv_cache, new_state_cache) = self._process_sequence(
            x, conv_cache, state_cache
        )

        if isinstance(cache, MambaCache):
            cache[0] = new_conv_cache
            cache[1] = new_state_cache

        return output


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mixer = MambaBlock(args)
        self.norm = nn.RMSNorm(args.hidden_size)

    def __call__(self, x: mx.array, cache):
        return self.mixer(self.norm(x), cache) + x


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [ResidualBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm_f = nn.RMSNorm(args.hidden_size)

    def __call__(self, x: mx.array, cache):
        x = self.embeddings(x)
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            x = layer(x, c)
        return self.norm_f(x)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.backbone = Mamba(args)
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
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
        return weights

    def make_cache(self):
        return [MambaCache() for _ in range(len(self.layers))]

    @property
    def layers(self):
        return self.backbone.layers
