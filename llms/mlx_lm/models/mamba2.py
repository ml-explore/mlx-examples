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
    dim: int = None
    intermediate_size: int = None
    time_step_limit: Tuple[float, float] = field(default_factory=lambda: (0.0, float("inf")))
    time_step_rank: Union[int, str] = "auto"
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    A_init_min: float = 1.0
    A_init_max: float = 16.0

    def __post_init__(self):
        if not hasattr(self, "intermediate_size"):
            self.intermediate_size = int(self.expand * self.hidden_size)
        if not hasattr(self, "hidden_size"):
            self.hidden_size = self.dim
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
        
        # Input projection
        d_in_proj = 2 * self.d_inner + 2 * self.n_groups * self.d_state + self.n_heads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=args.use_bias)
        
        # Improved initialization of dt
        dt = mx.exp(
            mx.random.uniform(
                low=math.log(args.time_step_min),
                high=math.log(args.time_step_max),
                shape=(self.n_heads,)
            )
        )

        dt = mx.clip(dt, args.time_step_floor, float('inf'))
        inv_dt = dt + mx.log(-mx.exp(-dt) + 1)  # Inverse softplus
        self.dt_bias = mx.array(inv_dt)

        # Improved A initialization
        A = mx.random.uniform(
            low=args.A_init_min,
            high=args.A_init_max,
            shape=(self.n_heads,)
        )
        self.A_log = mx.log(A)
        
        # Same D initialization
        self.D = mx.random.normal((self.n_heads,)) * args.initializer_range

        # Convolution with proper initialization
        self.conv1d = DepthWiseConv1d(
            channels=self.d_inner + 2 * self.n_groups * self.d_state,
            kernel_size=self.d_conv,
            bias=args.use_conv_bias,
            padding=self.d_conv-1
        )
        
        # Output projections
        self.norm = MambaRMSNormGated(self.d_inner, eps=args.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=args.use_bias)

    def __call__(self, u: mx.array, cache=None):
        batch_size, seq_len, _ = u.shape
        
        # Project input
        zxbcdt = self.in_proj(u)
        z = zxbcdt[..., :self.d_inner] 
        xBC = zxbcdt[..., self.d_inner:self.d_inner + (self.d_inner + 2 * self.n_groups * self.d_state)]
        dt = zxbcdt[..., -self.n_heads:]
        
        # Process dt
        dt = nn.softplus(dt + self.dt_bias)
        
        # Conv1d and activation
        xBC, conv_state = self.conv1d(xBC, cache[0] if cache else None)
        if cache is not None:
            cache[0] = conv_state
        xBC = silu(xBC)
        xBC = xBC[:, :seq_len, :]
        
        # Split conv output and reshape
        x = xBC[..., :self.d_inner]
        B = mx.reshape(xBC[..., self.d_inner:self.d_inner + self.n_groups * self.d_state], 
                    (batch_size, seq_len, self.n_groups, -1))
        C = mx.reshape(xBC[..., -self.n_groups * self.d_state:],
                    (batch_size, seq_len, self.n_groups, -1))
        
        x = mx.reshape(x, (batch_size, seq_len, self.n_heads, self.d_head))
        
        # Initialize state
        if cache and cache[1] is not None:
            prev_state = cache[1]
        else:
            prev_state = mx.zeros((batch_size, self.n_heads, self.d_head, self.d_state))
        
        # Compute dA
        A = -mx.exp(self.A_log)
        dt = mx.reshape(dt, (batch_size, seq_len, self.n_heads))
        dA = mx.exp(dt * mx.expand_dims(A, axis=(0, 1)))
        
        # Process sequence 
        next_state = prev_state
        outputs = []
        
        for t in range(seq_len):
            xt = x[:, t]
            Bt = B[:, t] 
            Ct = C[:, t]
            dAt = dA[:, t]
            
            # Update state
            dBx = mx.einsum('bh,bgd,bhp->bhpd', dAt, Bt, xt)
            next_state = next_state * mx.expand_dims(dAt, axis=(-1, -2)) + dBx
            
            # Compute output
            yt = mx.einsum('bhpd,bgd->bhp', next_state, Ct)
            yt = yt + xt * mx.expand_dims(self.D, -1)
            
            # Reshape and normalize
            yt = mx.reshape(yt, (batch_size, 1, self.d_inner))
            yt = self.norm(yt, z[:, t:t+1])
            outputs.append(self.out_proj(yt))
        
        # Update cache
        if cache is not None:
            cache[1] = next_state
            
        return mx.concatenate(outputs, axis=1)


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.residual_in_fp32 = args.residual_in_fp32
        self.mixer = Mamba2Block(args)
        self.norm = nn.RMSNorm(args.hidden_size)

    def __call__(self, x: mx.array, cache):
        if self.residual_in_fp32:
            x = x.astype(mx.float32)
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