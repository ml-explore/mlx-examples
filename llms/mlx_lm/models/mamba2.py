import math
from dataclasses import dataclass, field
from typing import Tuple, Union
import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from .cache import MambaCache

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
    

class Mamba2Block(nn.Module):
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

        # projection_size = 2 * args.intermediate_size + 2 * args.n_groups * args.state_size + args.num_heads
        projection_size = 2 * args.intermediate_size + 2 * args.state_size + args.num_heads
        self.in_proj = nn.Linear(
            args.hidden_size,
            projection_size,
            bias=args.use_bias
        )

        # self.conv_dim = args.intermediate_size + 2 * args.n_groups * args.state_size
        self.conv_dim = args.intermediate_size + 2 * args.state_size
        self.conv1d = DepthWiseConv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=args.conv_kernel,
            bias=args.use_conv_bias,
            groups=self.conv_dim,
            padding=args.conv_kernel - 1
        )

        self.A_log = mx.zeros(args.num_heads)
        self.D = mx.ones((args.num_heads,))
        self.dt_bias = mx.zeros(args.num_heads)

        self.out_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=args.use_bias)
        self.norm = MambaRMSNormGated(args.intermediate_size, eps=args.layer_norm_epsilon)

    def _ssd(self, x, A, B, C, chunk_size):
        batch, seq_len, nheads, head_dim = x.shape
        n_state = B.shape[-1]
        
        h = mx.zeros((batch, nheads, head_dim, n_state))
        ys = []
        
        for i in range(0, seq_len, chunk_size):
            chunk_size_i = min(chunk_size, seq_len - i)
            xi = x[:, i:i + chunk_size_i]
            Bi = B[:, i:i + chunk_size_i]
            Ci = C[:, i:i + chunk_size_i]
            
            for t in range(chunk_size_i):
                h = h * mx.exp(A)[:, None, None]
                h = h + mx.expand_dims(Bi[:, t], -2) * mx.expand_dims(xi[:, t], -1)
                y = mx.sum(h * mx.expand_dims(Ci[:, t], -2), axis=-1)
                ys.append(y)
        
        y = mx.stack(ys, axis=1)
        return y, h

    def __call__(self, x: mx.array, cache) -> mx.array:
        if cache is not None:
            return self.step(x, cache)

        A = -mx.exp(self.A_log)
        zxbcdt = self.in_proj(u)
        
        z, xBC, dt = mx.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            axis=-1,
        )
        
        dt = mx.softplus(dt + self.dt_bias)
        
        # Use the custom DepthWiseConv1d with cache
        xBC = self.conv1d(xBC, cache, cache_idx=0)
        xBC = mx.sigmoid(xBC) * xBC  # SiLU activation
        
        x, B, C = mx.split(
            xBC,
            [self.args.d_inner, self.args.d_state, self.args.d_state],
            axis=-1
        )
        
        x = self._reshape_heads(x, True)
        B = mx.expand_dims(B, axis=2)
        C = mx.expand_dims(C, axis=2)
        
        y, ssm_state = self._ssd(
            x * mx.expand_dims(dt, -1),
            A * dt,
            B,
            C,
            self.args.chunk_size
        )
        
        y = y + x * mx.expand_dims(self.D, -1)
        y = self._reshape_heads(y, False)
        y = self.norm(y, z)
        y = self.out_proj(y)

        if cache is not None:
            cache[1] = ssm_state

        return y

    def step(self, x: mx.array, cache) -> mx.array:
        """Single inference step"""
        assert x.shape[1] == 1, "Only one token can be decoded per inference step"
        
        zxbcdt = self.in_proj(mx.squeeze(x, 1))
        z, xBC, dt = mx.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            axis=-1,
        )

        # Use the custom DepthWiseConv1d with cache
        xBC = self.conv1d(xBC, cache, cache_idx=0)
        xBC = mx.sigmoid(xBC) * xBC  # SiLU activation

        x, B, C = mx.split(
            xBC,
            [self.args.d_inner, self.args.d_state, self.args.d_state],
            axis=-1
        )
        A = -mx.exp(self.A_log)

        dt = mx.softplus(dt + self.dt_bias)
        dA = mx.exp(dt * A)
        
        x = mx.reshape(x, (-1, self.args.nheads, self.args.headdim))
        
        ssm_state = cache[1]
        dBx = mx.expand_dims(dt, -1) * mx.expand_dims(B, 1) * mx.expand_dims(x, -1)
        ssm_state = ssm_state * mx.expand_dims(mx.expand_dims(dA, -1), -1) + dBx
        
        y = mx.sum(ssm_state * mx.expand_dims(mx.expand_dims(C, 1), 1), axis=-1)
        y = y + mx.expand_dims(self.D, -1) * x
        y = mx.reshape(y, (-1, self.args.nheads * self.args.headdim))
        
        y = self.norm(y, z)
        y = self.out_proj(y)

        # Update SSM state in cache
        cache[1] = ssm_state

        return mx.expand_dims(y, 1)


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mixer = Mamba2Block(args)
        self.norm = nn.RMSNorm(args.hidden_size)

    def __call__(self, x: mx.array, cache):
        return self.mixer(self.norm(x), cache) + x


class Mamba2(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [ResidualBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

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
        
        self.backbone = Mamba2(args)
        # self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

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

    def make_cache(self):
        return [MambaCache() for _ in range(len(self.layers))]

    @property
    def layers(self):
        return self.backbone.layers
