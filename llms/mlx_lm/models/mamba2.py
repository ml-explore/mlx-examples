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
    # Replace einsum operations with explicit reshape and matrix multiply
    batch, seqlen, nheads, dim = x.shape
    B = mx.expand_dims(B, axis=2)
    C = mx.expand_dims(C, axis=2)
    
    state = mx.zeros((batch, nheads, dim, B.shape[-1]))
    outputs = []
    
    for i in range(0, seqlen, chunk_size):
        chunk = slice(i, min(i + chunk_size, seqlen))
        dA = mx.exp(mx.expand_dims(A[chunk], axis=0))
        
        # Replace einsum with explicit operations
        x_chunk = x[:, chunk]  # [batch, chunk_size, nheads, dim]
        x_chunk = mx.transpose(x_chunk, [0, 2, 3, 1])  # [batch, nheads, dim, chunk_size]
        B_chunk = B[:, chunk]  # [batch, chunk_size, state_size]
        dBx = mx.matmul(x_chunk, B_chunk)  # [batch, nheads, dim, state_size]
        
        state = state * mx.expand_dims(dA, axis=-1) + dBx
        
        # Replace einsum with explicit operations
        C_chunk = C[:, chunk]  # [batch, chunk_size, state_size]
        y = mx.matmul(state, mx.transpose(C_chunk, [0, 2, 1]))  # [batch, nheads, dim, chunk_size]
        y = mx.transpose(y, [0, 3, 1, 2])  # [batch, chunk_size, nheads, dim]
        outputs.append(y)
    
    return mx.concatenate(outputs, axis=1), state
    

class DepthWiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, groups=None, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups if groups is not None else in_channels
        
        assert in_channels == out_channels, "In and out channels must be same for depthwise convolution"
        assert self.groups == in_channels, "Groups must be equal to in_channels for depthwise convolution"
        
        # Weight shape: (channels, 1, kernel_size) to match pretrained weights
        self.weight = mx.random.normal((in_channels, 1, kernel_size))
        self.bias = mx.zeros((out_channels,)) if bias else None

    def __call__(self, x: mx.array, cache=None, cache_idx: int = 0) -> mx.array:
        B, L, C = x.shape
        K = self.kernel_size

        # Validate input dimensions
        assert C == self.in_channels, f"Input channels {C} doesn't match expected {self.in_channels}"
        
        # Handle padding and caching
        if cache is not None:
            conv_states = cache[cache_idx]
            if conv_states is not None:
                # Validate cache shape
                assert conv_states.shape[0] == B, "Cache batch size mismatch"
                assert conv_states.shape[2] == C, "Cache channel count mismatch"
                x = mx.concatenate([conv_states, x], axis=1)
                L = x.shape[1]
        else:
            # Add left padding of size (kernel_size - 1)
            pad_left = K - 1
            x = mx.pad(x, [(0, 0), (pad_left, 0), (0, 0)])
            L = x.shape[1]

        # Pre-allocate output array if possible
        outputs = []
        
        # Process each channel independently
        for c in range(C):
            # Extract and prepare channel data
            x_c = x[:, :, c]  # Shape: [B, L]
            x_c = mx.expand_dims(x_c, axis=1)  # Shape: [B, 1, L]
            
            # Prepare filter weights
            w_c = self.weight[c]  # Get channel weights
            # Ensure filter is 3D: [depth(1), in_channels(1), kernel_size]
            if w_c.ndim == 2:
                w_c = mx.expand_dims(w_c, axis=0)
            elif w_c.ndim == 1:
                w_c = mx.expand_dims(mx.expand_dims(w_c, axis=0), axis=0)
            
            # Handle inference mode (single token)
            if L < K:
                pad_size = K - L
                x_c = mx.pad(x_c, [(0, 0), (0, 0), (pad_size, 0)])
            
            # Apply 1D convolution
            try:
                y_c = mx.conv_general(
                    x_c,
                    w_c,
                    stride=1,
                    padding=0  # Padding already handled
                )
                
                if self.bias is not None:
                    y_c = y_c + self.bias[c]
                
                # Remove singleton dimension and add to outputs
                outputs.append(mx.squeeze(y_c, axis=1))
                
            except Exception as e:
                raise RuntimeError(f"Convolution failed for channel {c}. Shapes: input={x_c.shape}, weight={w_c.shape}") from e

        # Stack channel outputs along last dimension
        y = mx.stack(outputs, axis=-1)  # Shape: [B, L', C]
        
        # Update cache if needed
        if cache is not None:
            # Store last (kernel_size - 1) tokens or entire input if shorter
            new_cache = x[:, -(K-1):, :] if L >= K else x
            cache[cache_idx] = new_cache

            if new_cache.shape != cache[cache_idx].shape:
                cache[cache_idx] = new_cache
                print(f"Cache updated at index {cache_idx}")
            else:
                print(f"Skipping cache update at index {cache_idx}, shapes are identical.")

        return y
    

class Mamba2Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        d_in_proj = 2 * args.intermediate_size + 2 * args.state_size + args.num_heads
        self.in_proj = nn.Linear(args.hidden_size, d_in_proj, bias=args.use_bias)

        conv_dim = args.intermediate_size + 2 * args.state_size
        self.conv1d = DepthWiseConv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.conv_kernel,
            groups=conv_dim,
            bias=args.use_conv_bias,
            padding=args.conv_kernel - 1
        )

        self.dt_bias = mx.random.normal((args.num_heads,)) * args.initializer_range
        self.A_log = mx.random.normal((args.num_heads,)) * args.initializer_range
        self.D = mx.random.normal((args.num_heads,)) * args.initializer_range

        self.norm = MambaRMSNormGated(args.intermediate_size, eps=args.layer_norm_epsilon)
        self.out_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=args.use_bias)

        if args.rescale_prenorm_residual:
            layer_scale = math.sqrt(1.0 / args.num_hidden_layers)
            self.out_proj.weight = self.out_proj.weight * layer_scale

    def __call__(self, x: mx.array, cache=None):
        # if cache is not None and self.args.use_cache:
        if cache is not None:
            return self.step(x, cache)

        # Calculate sizes
        d_model = self.args.intermediate_size
        d_state = self.args.state_size
        n_heads = self.args.num_heads
        
        # Compute A
        A = -mx.exp(self.A_log)
        
        # Project input
        zxbcdt = self.in_proj(x)
        
        # Correct splits for z, xBC, dt
        splits = [
            d_model,                # z
            d_model + 2 * d_state,  # xBC (delta, B, C concatenated)
            n_heads                 # dt
        ]
        
        # Split using cumulative indices
        z = zxbcdt[:, :, :splits[0]]
        xBC = zxbcdt[:, :, splits[0]:splits[0] + splits[1]]
        dt = zxbcdt[:, :, -splits[2]:]

        # Process dt
        dt = mx.clip(
            nn.softplus(dt + self.dt_bias),
            self.args.time_step_min,
            self.args.time_step_max
        )
        dt = mx.maximum(dt, self.args.time_step_floor)

        # Process convolution
        xBC = silu(self.conv1d(xBC))

        # Split convolved xBC into x, B, C
        x = xBC[:, :, :d_model]
        B = xBC[:, :, d_model:d_model + d_state]
        C = xBC[:, :, -d_state:]

        # Reshape for SSM computation
        b, l, hp = x.shape
        h = self.args.num_heads
        p = hp // h
        x = mx.reshape(x, (b, l, h, p))

        # Compute SSM
        y, ssm_state = ssd(
            x * mx.expand_dims(dt, -1),
            A * dt,
            B,
            C,
            self.args.chunk_size
        )

        # Add skip connection
        y = y + x * mx.expand_dims(self.D, -1)
        
        # Reshape back
        y = mx.reshape(y, (b, l, h * p))

        # Apply norm and projection
        y = self.norm(y + z)
        y = self.out_proj(y)

        # Update cache if needed
        if cache is not None and self.args.use_cache:
            cache[1] = ssm_state

        # Cast if needed
        if self.args.residual_in_fp32:
            y.astype(mx.float32)

        return y

    def step(self, u: mx.array, cache):
        batch_size = u.shape[0]
        seq_len = u.shape[1]
        outputs = []

        # Initialize SSM state if needed
        if cache[1] is None:
            cache[1] = mx.zeros((
                batch_size,
                self.args.num_heads,
                self.args.head_dim,
                self.args.state_size
            ))

        for pos in range(seq_len):
            # Get single token
            u_t = u[:, pos:pos+1, :]

            # Project input
            zxbcdt = self.in_proj(u_t)
            
            # Calculate sizes
            d_model = self.args.intermediate_size
            d_state = self.args.state_size
            n_heads = self.args.num_heads
            d_head = self.args.head_dim
            
            # Split projected input
            # conv_dim = d_model + 2 * d_state (this should match self.conv1d.in_channels)
            z = zxbcdt[:, :, :d_model]
            xBC = zxbcdt[:, :, d_model:d_model + 2*d_state + d_model]  # Include the full conv dimension
            dt = zxbcdt[:, :, -(n_heads):]
            
            # Process dt
            dt = mx.reshape(dt, (batch_size, n_heads))
            dt = mx.clip(
                nn.softplus(dt + self.dt_bias),
                self.args.time_step_min,
                self.args.time_step_max
            )
            dt = mx.maximum(dt, self.args.time_step_floor)

            # Process convolution with correct dimensions
            xBC = self.conv1d(xBC, cache=cache, cache_idx=0)
            xBC = silu(xBC)

            # Split convolved xBC into x, B, C with correct dimensions
            x = xBC[:, :, :d_model]
            B = xBC[:, :, d_model:d_model + d_state]
            C = xBC[:, :, -d_state:]
            
            # Reshape tensors for SSM computation
            x = mx.reshape(x, (batch_size, 1, n_heads, d_head))
            x = mx.squeeze(x, axis=1)  # (batch, heads, dim)
            
            B = mx.reshape(B, (batch_size, 1, d_state))
            B = mx.broadcast_to(B, (batch_size, n_heads, d_state))
            B = mx.expand_dims(B, axis=2)  # (batch, heads, 1, state)
            
            C = mx.reshape(C, (batch_size, 1, d_state))
            C = mx.broadcast_to(C, (batch_size, n_heads, d_state))
            C = mx.expand_dims(C, axis=3)  # (batch, heads, state, 1)

            # Compute SSM updates
            A = -mx.exp(self.A_log)
            dA = mx.exp(dt * mx.expand_dims(A, 0))
            dA = mx.expand_dims(mx.expand_dims(dA, -1), -1)  # (batch, heads, 1, 1)

            # Update state with proper shapes
            x = mx.expand_dims(x, axis=3)  # (batch, heads, dim, 1)
            dBx = mx.matmul(x, B)  # (batch, heads, dim, state)
            
            ssm_state = cache[1]
            ssm_state = ssm_state * dA + dBx
            cache[1] = ssm_state

            # Compute output
            y = mx.matmul(ssm_state, C)  # (batch, heads, dim, 1)
            y = mx.squeeze(y, axis=-1)  # (batch, heads, dim)
            
            # Add skip connection
            y = y + x[:, :, :, 0] * mx.expand_dims(self.D, -1)
            
            # Reshape and process output
            y = mx.reshape(y, (batch_size, 1, n_heads * d_head))
            y = self.norm(y + z)
            y = self.out_proj(y)

            if self.args.residual_in_fp32:
                y = y.astype(mx.float32)

            outputs.append(y)

        return mx.concatenate(outputs, axis=1)


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

        print('ouput')
        return logits

    def make_cache(self, batch_size=1):
        return [Mamba2Cache(batch_size, self.args.num_heads, self.args.head_dim, self.args.state_size) for _ in range(len(self.layers))]
    
    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if "conv1d.weight" in k:
                # Ensure weights are in correct shape (channels, 1, kernel_size)
                if v.ndim == 2:
                    v = mx.expand_dims(v, axis=1)
                elif v.ndim == 1:
                    v = mx.expand_dims(mx.expand_dims(v, axis=0), axis=0)
                sanitized[k] = v
            else:
                sanitized[k] = v
        return sanitized

    @property
    def layers(self):
        return self.backbone.layers
