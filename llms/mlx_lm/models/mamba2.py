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
    rms_norm: bool
    chunk_size: int
    tie_word_embeddings: bool
    intermediate_size: int = None
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

def ssd(x, A, B, C, chunk_size):
    batch, seqlen, nheads, dim = x.shape
    B = mx.expand_dims(B, axis=2)
    C = mx.expand_dims(C, axis=2)
    
    state = mx.zeros((batch, nheads, dim, B.shape[-1]))
    outputs = []
    
    for i in range(0, seqlen, chunk_size):
        chunk = slice(i, min(i + chunk_size, seqlen))
        dA = mx.exp(mx.expand_dims(A[chunk], axis=0))
        
        x_chunk = x[:, chunk]  # [batch, chunk_size, nheads, dim]
        x_chunk = mx.transpose(x_chunk, [0, 2, 3, 1])  # [batch, nheads, dim, chunk_size]
        B_chunk = B[:, chunk]  # [batch, chunk_size, state_size]
        dBx = mx.matmul(x_chunk, B_chunk)  # [batch, nheads, dim, state_size]
        
        state = state * mx.expand_dims(dA, axis=-1) + dBx
        
        C_chunk = C[:, chunk]  # [batch, chunk_size, state_size]
        y = mx.matmul(state, mx.transpose(C_chunk, [0, 2, 1]))  # [batch, nheads, dim, chunk_size]
        y = mx.transpose(y, [0, 3, 1, 2])  # [batch, chunk_size, nheads, dim]
        outputs.append(y)
    
    return mx.concatenate(outputs, axis=1), state
    

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

        # Adjust the weight tensor to match the input channels
        if C != self.channels:
            adjusted_weight = self.weight[:C, :, :]
        else:
            adjusted_weight = self.weight

        y = mx.conv_general(x, adjusted_weight, groups=C)
        
        if self.bias is not None:
            # Adjust the bias to match the input channels
            adjusted_bias = self.bias[:C] if C != self.channels else self.bias
            y = y + adjusted_bias

        return y, x[:, -K + 1:, :]



class Mamba2Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        # Calculate dimensions
        self.d_model = args.hidden_size
        self.d_state = args.state_size
        self.d_conv = args.conv_kernel
        self.expand = args.expand
        if args.intermediate_size == None:
            self.d_inner = int(self.expand * self.d_model)
        else:
            self.d_inner = args.intermediate_size
        self.n_groups = args.n_groups
        self.n_heads = args.num_heads
        self.d_head = self.d_inner // self.n_heads
        
        # Input projection
        d_in_proj = 2 * self.d_inner + 2 * self.n_groups * self.d_state + self.n_heads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=args.use_bias)
        
        # Convolution
        conv_dim = self.d_inner + 2 * self.n_groups * self.d_state
        self.conv1d = DepthWiseConv1d(
            channels=conv_dim,
            kernel_size=self.d_conv,
            bias=args.use_conv_bias
        )
        
        # SSM parameters
        self.dt_bias = mx.random.normal((self.n_heads,)) * args.initializer_range
        self.A_log = mx.random.normal((self.n_heads,)) * args.initializer_range
        self.D = mx.random.normal((self.n_heads,)) * args.initializer_range
        
        # Output projection
        self.norm = MambaRMSNormGated(self.d_inner, eps=args.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=args.use_bias)
        
        if args.rescale_prenorm_residual:
            layer_scale = math.sqrt(1.0 / args.num_hidden_layers)
            self.out_proj.weight = self.out_proj.weight * layer_scale

    def __call__(self, u: mx.array, cache=None):
        batch_size, seq_len, _ = u.shape
        
        # Project input
        proj = self.in_proj(u)  # [batch, seq_len, d_in_proj]
        
        # Calculate split indices and slice tensors
        z = proj[..., :self.d_inner]
        x_conv = proj[..., self.d_inner:self.d_inner + (self.d_inner + 2 * self.d_state)]
        dt = proj[..., -self.n_heads:]
        
        # Process time steps
        dt = nn.softplus(dt + self.dt_bias)
        dt = mx.clip(dt, self.args.time_step_min, self.args.time_step_max)
        dt = mx.maximum(dt, self.args.time_step_floor)
        
        # Convolution and activation
        x_conv, conv_state = self.conv1d(x_conv, cache[0] if cache else None)
        if cache is not None:
            cache[0] = conv_state
        x_conv = silu(x_conv)
        
        # Split conv output
        x = x_conv[..., :self.d_inner]
        B = x_conv[..., self.d_inner:self.d_inner + self.d_state]
        C = x_conv[..., -self.d_state:]
        
        # Reshape x for SSM
        x = mx.reshape(x, (batch_size, seq_len, self.n_heads, self.d_head))
        
        # Process B and C without reshaping heads
        B = mx.expand_dims(B, axis=2)  # [batch, seq_len, 1, d_state]
        B = mx.broadcast_to(B, (batch_size, seq_len, self.n_heads, self.d_state))
        
        C = mx.expand_dims(C, axis=2)  # [batch, seq_len, 1, d_state]
        C = mx.broadcast_to(C, (batch_size, seq_len, self.n_heads, self.d_state))
        
        # Initialize or get previous state
        if cache and cache[1] is not None:
            prev_state = cache[1]
        else:
            prev_state = mx.zeros((batch_size, self.n_heads, self.d_head, self.d_state))
        
        # Compute dA
        dA = -mx.exp(self.A_log)  # [n_heads]
        dt = mx.reshape(dt, (batch_size, seq_len, self.n_heads))  # Ensure correct shape
        dA = mx.exp(mx.expand_dims(dt * mx.expand_dims(dA, 0), -1))  # [batch, seq_len, n_heads, 1]
        dA = mx.expand_dims(dA, -1)  # [batch, seq_len, n_heads, 1, 1]
        
        # Process sequence
        next_state = prev_state
        outputs = []
        
        for t in range(seq_len):
            # Get current step tensors
            xt = x[:, t]  # [batch, n_heads, d_head]
            Bt = B[:, t]  # [batch, n_heads, d_state]
            Ct = C[:, t]  # [batch, n_heads, d_state]
            dAt = dA[:, t]  # [batch, n_heads, 1, 1]
            
            # Update state
            next_state = (
                next_state * dAt +  # Broadcasting: [batch, n_heads, d_head, d_state] * [batch, n_heads, 1, 1]
                mx.matmul(
                    mx.expand_dims(xt, -1),  # [batch, n_heads, d_head, 1]
                    mx.expand_dims(Bt, -2)   # [batch, n_heads, 1, d_state]
                )
            )
            
            # Compute output
            yt = mx.matmul(
                next_state,  # [batch, n_heads, d_head, d_state]
                mx.expand_dims(Ct, -1)  # [batch, n_heads, d_state, 1]
            )
            yt = mx.squeeze(yt, -1)  # [batch, n_heads, d_head]
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