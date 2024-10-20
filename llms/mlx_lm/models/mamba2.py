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

        self.conv_dim = args.intermediate_size + 2 * args.n_groups * args.state_size
        self.conv1d = DepthWiseConv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=args.conv_kernel,
            bias=args.use_conv_bias,
            groups=self.conv_dim,
            padding=args.conv_kernel - 1
        )

        projection_size = args.intermediate_size + self.conv_dim + args.num_heads
        self.in_proj = nn.Linear(
            args.hidden_size,
            projection_size,
            bias=args.use_bias
        )

        self.act = nn.SiLU()

        self.A_log = mx.zeros(args.num_heads)
        self.D = mx.ones((args.num_heads,))
        self.dt_bias = mx.zeros(args.num_heads)

        self.out_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=args.use_bias)
        self.norm = MambaRMSNormGated(args.intermediate_size, eps=args.layer_norm_epsilon)

    def ssm_step(self, x, state, dt_proj):
        A = -mx.exp(self.A_log)
        D = self.D
        delta = nn.softplus(dt_proj + self.dt_bias)
        
        B, C = mx.split(x, indices_or_sections=[self.state_size * self.n_groups], axis=-1)
        
        batch_size = B.shape[0]
        B = B.reshape(batch_size, self.n_groups, self.state_size)
        C = C.reshape(batch_size, -1, self.state_size)
        
        delta = delta.reshape(batch_size, self.num_heads, 1)
        A = A.reshape(1, self.num_heads, 1)
        
        if state is None:
            new_state = delta * B
        else:
            new_state = delta * (B + state * mx.exp(delta * A))
        
        y = mx.sum(new_state[:, :, None, :] * C[:, None, :, :], axis=(-1, -2))
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

            # Use the new DepthWiseConv1d with caching
            conv_out, cache[0] = self.conv1d(mx.expand_dims(x_t, 1), cache[0])
            x_t = conv_out.squeeze(1)
            x_t = nn.silu(x_t)
            y_t, cache[1] = self.ssm_step(x_t, cache[1], dt_proj)
            z_t = nn.silu(z_t)
            
            # Element-wise multiplication
            output_t = y_t[:, :, None] * z_t[:, None, :]
            
            # Sum across the second dimension to match the intermediate_size
            output_t = output_t.sum(axis=1)
            
            output_t = self.out_proj(output_t)
            outputs.append(output_t)
        
        output = mx.stack(outputs, axis=1)
        return output


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
