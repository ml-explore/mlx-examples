import math
from dataclasses import dataclass, field
from typing import Tuple, Union
import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from .cache import Mamba2Cache


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
    rms_norm: bool
    chunk_size: int
    tie_word_embeddings: bool
    use_cache: bool = True
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


def silu(x):
    return x * mx.sigmoid(x)


class MambaRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps

    def __call__(self, hidden_states, gate=None):
        # Fuse operations where possible
        if gate is not None:
            hidden_states = hidden_states * nn.silu(gate)
        # Compute variance in fp32 for better numerical stability
        hidden_states_fp32 = hidden_states.astype(mx.float32)
        variance = mx.mean(hidden_states_fp32 * hidden_states_fp32, axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


def ssd_optimized(x, A, B, C, chunk_size):
    batch, seqlen, nheads, dim = x.shape
    B = mx.expand_dims(B, axis=2)
    C = mx.expand_dims(C, axis=2)
    
    output = mx.zeros((batch, seqlen, nheads, dim))
    state = mx.zeros((batch, nheads, dim, B.shape[-1]))
    
    for i in range(0, seqlen, chunk_size):
        chunk = slice(i, min(i + chunk_size, seqlen))
        chunk_size_actual = min(chunk_size, seqlen - i)
        
        dA = mx.exp(mx.expand_dims(A[chunk], axis=0))
        x_chunk = mx.transpose(x[:, chunk], [0, 2, 3, 1])
        dBx = mx.matmul(x_chunk, B[:, chunk])
        state = state * mx.expand_dims(dA, axis=-1) + dBx
        y = mx.matmul(state, mx.transpose(C[:, chunk], [0, 2, 1]))
        output[:, i:i+chunk_size_actual] = mx.transpose(y, [0, 3, 1, 2])
    
    return output, state


def update_conv_cache(x: mx.array, cache, kernel_size: int) -> Tuple[mx.array, mx.array]:
    """Update convolution cache for sequential processing."""
    B, L, C = x.shape
    
    if cache is None:
        # Initialize cache with zeros
        cache = mx.zeros((B, kernel_size - 1, C))
    
    # Concatenate cache with current input
    x_with_cache = mx.concatenate([cache, x], axis=1)
    
    # Update cache with the last (kernel_size - 1) elements
    new_cache = x_with_cache[:, -kernel_size+1:] if x_with_cache.shape[1] >= kernel_size else x_with_cache
    
    return x_with_cache, new_cache


class Mamba2Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.intermediate_size = int(args.expand * args.hidden_size)
        self.state_size = args.state_size
        self.num_heads = args.num_heads
        self.head_dim = args.head_dim
        self.ssm_state_size = args.state_size
        self.n_groups = args.n_groups
        self.conv_kernel = args.conv_kernel

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        
        self.in_proj = nn.Linear(args.hidden_size, projection_size, bias=args.use_bias)
        
        # Using built-in Conv1d instead of custom DepthwiseConv1d
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=args.conv_kernel,
            groups=self.conv_dim,  # For depthwise convolution
            padding=0,  # We'll handle padding manually with the cache
            bias=args.use_conv_bias
        )

        self.dt_bias = mx.random.normal((args.num_heads,)) * args.initializer_range
        self.A_log = mx.random.normal((args.num_heads,)) * args.initializer_range
        self.D = mx.random.normal((args.num_heads,)) * args.initializer_range

        self.norm = MambaRMSNormGated(args.intermediate_size, eps=args.layer_norm_epsilon)
        self.out_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=args.use_bias)

        if args.rescale_prenorm_residual:
            layer_scale = math.sqrt(1.0 / args.num_hidden_layers)
            self.out_proj.weight = self.out_proj.weight * layer_scale

    def __call__(self, u: mx.array, cache):
        batch_size, seq_len, _ = u.shape
        
        projected = self.in_proj(u)
        d_conv = self.conv_dim
        
        z = projected[..., :self.intermediate_size]
        xBC = projected[..., self.intermediate_size:self.intermediate_size + d_conv]
        dt = projected[..., -self.num_heads:]
        
        dt = mx.clip(
            nn.softplus(dt + self.dt_bias),
            self.args.time_step_min,
            self.args.time_step_max
        )
        dt = mx.maximum(dt, self.args.time_step_floor)
        
        # Handle convolution with separate cache update
        if cache is not None:
            # Update cache and get padded input
            xBC_padded, new_cache = update_conv_cache(xBC, cache.conv_states, self.conv_kernel)
            cache.conv_states = new_cache
            
            # Prepare input for conv1d: [B, L, C] -> [B, C, L]
            xBC_conv = mx.transpose(xBC_padded, [0, 2, 1])
            
            # Apply convolution
            xBC = self.conv1d(xBC_conv)
            
            # Transform back: [B, C, L] -> [B, L, C]
            xBC = mx.transpose(xBC, [0, 2, 1])
            
            # Take only the relevant part corresponding to input length
            xBC = xBC[:, :seq_len]
        else:
            # For training, use regular convolution with padding
            xBC = mx.transpose(xBC, [0, 2, 1])
            xBC = self.conv1d(xBC)
            xBC = mx.transpose(xBC, [0, 2, 1])
        
        xBC = silu(xBC)
        
        x = xBC[..., :self.intermediate_size]
        BC = xBC[..., self.intermediate_size:]
        B = BC[..., :self.state_size]
        C = BC[..., self.state_size:]
        
        x = mx.reshape(x, (-1, seq_len, self.num_heads, self.intermediate_size // self.num_heads))
        
        A = -mx.exp(self.A_log)
        D_expanded = mx.expand_dims(self.D, -1)
        
        if cache is not None and cache.ssm_state is None:
            cache.ssm_state = mx.zeros((
                batch_size,
                self.num_heads,
                self.intermediate_size // self.num_heads,
                self.state_size
            ))
        
        if cache is not None:
            output = mx.zeros((batch_size, seq_len, self.args.hidden_size))
            
            for pos in range(seq_len):
                x_t = x[:, pos:pos+1]
                
                dA = mx.exp(dt[:, pos:pos+1] * mx.expand_dims(A, 0))
                dA = mx.expand_dims(mx.expand_dims(dA, -1), -1)
                
                x_expanded = mx.expand_dims(x_t, axis=3)
                dBx = mx.matmul(x_expanded, mx.expand_dims(B[:, pos:pos+1], axis=2))
                
                cache.ssm_state = cache.ssm_state * dA + dBx
                
                y = mx.matmul(cache.ssm_state, mx.expand_dims(C[:, pos:pos+1], axis=3))
                y = mx.squeeze(y, axis=-1)
                y = y + x_t * D_expanded
                
                y = mx.reshape(y, (batch_size, 1, -1))
                y = self.norm(y + z[:, pos:pos+1])
                y = self.out_proj(y)
                
                if self.args.residual_in_fp32:
                    y = y.astype(mx.float32)
                
                output = output.at[:, pos:pos+1].set(y)
        else:
            y, ssm_state = ssd_optimized(
                x * mx.expand_dims(dt, -1),
                -mx.exp(self.A_log) * dt,
                B, C,
                self.args.chunk_size
            )
            
            y = mx.reshape(
                y + x * mx.expand_dims(self.D, -1),
                (batch_size, seq_len, -1)
            )
            
            y = self.norm(y + z)
            output = self.out_proj(y)
            
            if self.args.residual_in_fp32:
                output = output.astype(mx.float32)
        
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

    def make_cache(self, batch_size=1):
        return [Mamba2Cache() for _ in range(len(self.layers))]
    
    def sanitize(self, weights):
        for k, v in weights.items():
            if "conv1d.weight" in k and v.ndim == 3:
                weights[k] = v.moveaxis(2, 1)
        return weights

    @property
    def layers(self):
        return self.backbone.layers