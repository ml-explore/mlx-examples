import math
from dataclasses import dataclass, field
from typing import Tuple, Union
import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from .cache import MambaCache

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
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
    rescale_prenorm_residual: bool
    rms_norm: bool
    chunk_size: int
    tie_word_embeddings: bool
    time_step_limit: Tuple[float, float]
    time_step_rank: Union[int, str]
    time_step_min: float
    time_step_max: float
    time_step_floor: float
    norm_before_gate: bool = True

    def __post_init__(self):
        if not hasattr(self, "intermediate_size"):
            self.intermediate_size = int(self.expand * self.hidden_size)
        if not hasattr(self, "head_dim"):
            self.head_dim = self.hidden_size // self.num_heads
        if self.time_step_rank == "auto":
            self.time_step_rank = math.ceil(self.hidden_size / 16)


def silu(x):
    return x * mx.sigmoid(x)


class DepthWiseConv1d(nn.Module):
    def __init__(self, channels, kernel_size, bias=True, padding=0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight = mx.random.normal((channels, kernel_size, 1))
        self.bias = mx.zeros((channels,)) if bias else None

    def __call__(self, x, cache=None):
        B, L, C = x.shape
        _, K, _ = self.weight.shape

        if cache is not None:
            x = mx.concatenate([cache, x], axis=1)
        else:
            x = mx.pad(x, [(0, 0), (K - 1, 0), (0, 0)])

        y = mx.conv_general(x, self.weight, groups=C)
        y = y + self.bias
        return y, x[:, -K + 1:, :]


def ssd_forward_attn(
    x: mx.array,
    dt: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt_bias: mx.array,
    dt_min: float,
    dt_max: float,
) -> Tuple[mx.array, mx.array]:
    b, l, h, dh = x.shape
    _, _, g, _ = B.shape

    if dt_bias is not None:
        dt = dt + dt_bias.reshape(1, 1, -1)

    dt = nn.softplus(dt)
    dt = mx.clip(dt, a_min=dt_min, a_max=dt_max)

    B = mx.swapaxes(mx.swapaxes(B, 1, 3), 1, 2)
    C = mx.swapaxes(C, 1, 2)

    CB = C @ B
    CB = mx.repeat(CB, repeats=h // g, axis=1)

    dtA = dt * A.reshape(1, 1, -1)
    dtA = mx.swapaxes(dtA, 1, 2)

    decay = mx.exp(segsum(dtA))

    surrogate_attention_matrix = mx.tril(CB * decay, 0)

    dtx = dt.reshape(b, l, h, 1) * x
    y = surrogate_attention_matrix @ dtx.swapaxes(1, 2)
    y = mx.swapaxes(y, 1, 2)

    decay = decay[:, :, -1, :].reshape(b, h, l).swapaxes(1, 2).reshape(b, l, h, 1)
    B = mx.repeat(B, h // g, axis=1).swapaxes(2, 3)
    dtxdecay = dtx * decay
    dtxdecay = dtxdecay.swapaxes(1, 2).swapaxes(2, 3)
    next_state = dtxdecay @ B

    if D is not None:
        y += x * D.reshape(1, 1, h, 1)

    y = y.reshape(b, l, h * dh)

    return y, next_state


def segsum(x):
    l = x.shape[-1]
    x = mx.repeat(x[..., None], l, axis=-1)
    x = mx.tril(x, -1)
    x_segsum = mx.cumsum(x, axis=-2)
    return x_segsum


class Mamba2Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        # Same dimensions as before
        self.d_model = args.hidden_size
        self.d_state = args.state_size
        self.d_conv = args.conv_kernel
        self.expand = args.expand
        self.d_inner = int(self.expand * self.d_model)
        self.n_groups = args.n_groups
        self.n_heads = args.num_heads
        self.d_head = self.d_inner // self.n_heads
        self.chunk_size = args.chunk_size
        
        # Input projection
        d_in_proj = 2 * self.d_inner + 2 * self.n_groups * self.d_state + self.n_heads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=args.use_bias)

        # Parameters
        self.dt_bias = mx.random.normal((self.n_heads,)) * args.initializer_range
        self.A_log = mx.random.normal((self.n_heads,)) * args.initializer_range
        self.D = mx.random.normal((self.n_heads,)) * args.initializer_range

        # Convolution
        self.conv1d = DepthWiseConv1d(
            channels=self.d_inner + 2 * self.n_groups * self.d_state,
            kernel_size=self.d_conv,
            bias=args.use_conv_bias,
            padding=self.d_conv-1
        )
        
        # Output projections
        self.norm = nn.RMSNorm(self.d_inner, eps=args.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=args.use_bias)

    def __call__(self, u: mx.array, cache=None):
        batch_size, seq_len, _ = u.shape
        
        # Get or initialize states from cache
        if cache is None:
            cache = [None, None]  # [conv_state, ssm_state]
        conv_state, _ = cache  # We ignore ssm_state as it's not used in the parallel version

        # Project input
        zxBCdt = self.in_proj(u)

        # Split projections
        z, xBC, dt = mx.split(
            zxBCdt, 
            [self.d_inner, 2 * self.d_inner + 2 * self.n_groups * self.d_state], 
            axis=-1
        )

        # Process convolution
        xBC, conv_state = self.conv1d(xBC, conv_state)
        xBC = silu(xBC)
        xBC = xBC[:, :seq_len, :]

        # Split conv output
        x, B, C = mx.split(
            xBC, 
            [self.d_inner, self.d_inner + self.d_state * self.n_groups], 
            axis=-1
        )

        # Reshape for SSM processing
        x = mx.reshape(x, (batch_size, seq_len, self.n_heads, self.d_head))
        B = mx.reshape(B, (batch_size, seq_len, self.n_groups, -1))
        C = mx.reshape(C, (batch_size, seq_len, self.n_groups, -1))

        # Process with parallel attention
        A = -mx.exp(self.A_log)
        y, next_state = ssd_forward_attn(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=self.D,
            dt_bias=self.dt_bias,
            dt_min=self.args.time_step_min,
            dt_max=self.args.time_step_max
        )

        # Apply normalization based on norm_before_gate setting
        if self.args.norm_before_gate:
            y = self.norm(y)
            y = y * nn.silu(z)
        else:
            y = y * nn.silu(z)
            y = self.norm(y)

        # Final projection
        y = self.out_proj(y)

        # Update cache
        cache[0] = conv_state
        cache[1] = next_state

        return y


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.residual_in_fp32 = args.residual_in_fp32
        self.mixer = Mamba2Block(args)
        self.norm = nn.RMSNorm(args.hidden_size)

    def __call__(self, x: mx.array, cache):
        # if self.residual_in_fp32:
        #     x = x.astype(mx.float32)
        normed = self.norm(x)
        output = self.mixer(normed, cache)
        return output + x


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
        
        hidden = x
        for layer, c in zip(self.layers, cache):
            hidden = layer(hidden, c)
        return self.norm_f(hidden)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.backbone = Mamba2(args)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        hidden = self.backbone(inputs, cache)
        
        if self.args.tie_word_embeddings:
            logits = self.backbone.embeddings.as_linear(hidden)
        else:
            logits = self.lm_head(hidden)
        
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