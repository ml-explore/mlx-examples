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

    def ssm_step(self, x, state, dt):
        A = -mx.exp(self.A_log)
        D = self.D
        dt = nn.softplus(dt + self.dt_bias)
        
        B, C = mx.split(x, indices_or_sections=[self.state_size * self.n_groups], axis=-1)
        
        batch_size = B.shape[0]
        B = B.reshape(batch_size, self.n_groups, self.state_size)
        C = C.reshape(batch_size, -1, self.state_size)
        
        dt = dt.reshape(batch_size, self.num_heads, 1)
        A = A.reshape(1, self.num_heads, 1)
        
        if state is None:
            new_state = dt * B
        else:
            new_state = dt * (B + state * mx.exp(dt * A))
        
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
            zxbcdt = self.in_proj(xt)
            
            z, xBC, dt = mx.split(
                zxbcdt,
                # indices_or_sections=[self.conv_dim, self.conv_dim + self.intermediate_size],
                indices_or_sections=[
                    self.intermediate_size,
                    self.intermediate_size + 2 * self.state_size,
                    self.num_heads
                ],
                axis=-1
            )

            # Use the new DepthWiseConv1d with caching
            conv_out, cache[0] = self.conv1d(mx.expand_dims(z, 1), cache[0])
            z = conv_out.squeeze(1)
            z = nn.silu(z)
            y_t, cache[1] = self.ssm_step(z, cache[1], dt)
            xBC = nn.silu(xBC)
            
            # Element-wise multiplication
            output_t = y_t[:, :, None] * xBC[:, None, :]
            
            output_t = self.norm(output_t)
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





# ------



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
        
        self.weight = mx.random.normal((in_channels, 1, kernel_size))
        self.bias = mx.zeros((out_channels,)) if bias else None

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        B, L, C = x.shape
        K = self.kernel_size

        assert C == self.in_channels, f"Input channels {C} doesn't match expected {self.in_channels}"
        
        if cache is not None and 'conv_states' in cache:
            conv_states = cache['conv_states']
            if conv_states is not None:
                assert conv_states.shape[0] == B, "Cache batch size mismatch"
                assert conv_states.shape[2] == C, "Cache channel count mismatch"
                x = mx.concatenate([conv_states, x], axis=1)
            
        # Process each channel independently
        outputs = []
        for c in range(C):
            x_c = x[:, :, c]
            x_c = mx.expand_dims(x_c, axis=1)
            
            w_c = self.weight[c]
            if w_c.ndim == 2:
                w_c = mx.expand_dims(w_c, axis=0)
            elif w_c.ndim == 1:
                w_c = mx.expand_dims(mx.expand_dims(w_c, axis=0), axis=0)
            
            # Apply convolution
            y_c = mx.conv_general(
                x_c,
                w_c,
                stride=1,
                padding=0
            )
            
            if self.bias is not None:
                y_c = y_c + self.bias[c]
            
            outputs.append(mx.squeeze(y_c, axis=1))

        y = mx.stack(outputs, axis=-1)
        
        # Update cache
        if cache is not None:
            cache['conv_states'] = x[:, -K+1:, :] if x.shape[1] >= K else x
            
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
        if cache is not None:
            return self.step(x, cache)

        # Regular forward pass code remains the same...
        d_model = self.args.intermediate_size
        d_state = self.args.state_size
        n_heads = self.args.num_heads
        
        A = -mx.exp(self.A_log)
        zxbcdt = self.in_proj(x)
        
        splits = [d_model, d_model + 2 * d_state, n_heads]
        z = zxbcdt[:, :, :splits[0]]
        xBC = zxbcdt[:, :, splits[0]:splits[0] + splits[1]]
        dt = zxbcdt[:, :, -splits[2]:]

        dt = mx.clip(
            nn.softplus(dt + self.dt_bias),
            self.args.time_step_min,
            self.args.time_step_max
        )
        dt = mx.maximum(dt, self.args.time_step_floor)

        xBC = silu(self.conv1d(xBC))

        x = xBC[:, :, :d_model]
        B = xBC[:, :, d_model:d_model + d_state]
        C = xBC[:, :, -d_state:]

        b, l, hp = x.shape
        h = self.args.num_heads
        p = hp // h
        x = mx.reshape(x, (b, l, h, p))

        y, ssm_state = ssd(x * mx.expand_dims(dt, -1), A * dt, B, C, self.args.chunk_size)
        y = y + x * mx.expand_dims(self.D, -1)
        y = mx.reshape(y, (b, l, h * p))

        y = self.norm(y + z)
        y = self.out_proj(y)

        if self.args.residual_in_fp32:
            y = y.astype(mx.float32)

        return y

    def step(self, u: mx.array, cache):
        batch_size = u.shape[0]
        seq_len = u.shape[1]
        outputs = []

        # Initialize cache if needed
        if cache.conv_states is None:
            conv_dim = self.args.intermediate_size + 2 * self.args.state_size
            cache.conv_states = mx.zeros((
                batch_size,
                self.args.conv_kernel - 1,
                conv_dim
            ))
            
        if cache.ssm_state is None:
            cache.ssm_state = mx.zeros((
                batch_size,
                self.args.num_heads,
                self.args.head_dim,
                self.args.state_size
            ))

        for pos in range(seq_len):
            u_t = u[:, pos:pos+1, :]
            zxbcdt = self.in_proj(u_t)
            
            d_model = self.args.intermediate_size
            d_state = self.args.state_size
            n_heads = self.args.num_heads
            
            z = zxbcdt[:, :, :d_model]
            xBC = zxbcdt[:, :, d_model:d_model + 2*d_state + d_model]
            dt = zxbcdt[:, :, -(n_heads):]
            
            dt = mx.reshape(dt, (batch_size, n_heads))
            dt = mx.clip(
                nn.softplus(dt + self.dt_bias),
                self.args.time_step_min,
                self.args.time_step_max
            )
            dt = mx.maximum(dt, self.args.time_step_floor)

            # Create a temporary cache dictionary for the convolution
            conv_cache = {'conv_states': cache.conv_states}
            xBC = self.conv1d(xBC, cache=conv_cache)
            cache.conv_states = conv_cache['conv_states']
            
            xBC = silu(xBC)

            x = xBC[:, :, :d_model]
            B = xBC[:, :, d_model:d_model + d_state]
            C = xBC[:, :, -d_state:]
            
            x = mx.reshape(x, (batch_size, 1, n_heads, self.args.head_dim))
            x = mx.squeeze(x, axis=1)

            B = mx.reshape(B, (batch_size, 1, d_state))
            B = mx.broadcast_to(B, (batch_size, n_heads, d_state))
            B = mx.expand_dims(B, axis=2)

            C = mx.reshape(C, (batch_size, 1, d_state))
            C = mx.broadcast_to(C, (batch_size, n_heads, d_state))
            C = mx.expand_dims(C, axis=3)

            A = -mx.exp(self.A_log)
            dA = mx.exp(dt * mx.expand_dims(A, 0))
            dA = mx.expand_dims(mx.expand_dims(dA, -1), -1)

            x = mx.expand_dims(x, axis=3)
            dBx = mx.matmul(x, B)
            
            cache.ssm_state = cache.ssm_state * dA + dBx

            y = mx.matmul(cache.ssm_state, C)
            y = mx.squeeze(y, axis=-1)
            
            y = y + x[:, :, :, 0] * mx.expand_dims(self.D, -1)
            
            y = mx.reshape(y, (batch_size, 1, n_heads * self.args.head_dim))
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

        return logits

    def make_cache(self):
        return [Mamba2Cache() for _ in range(len(self.layers))]
    
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
