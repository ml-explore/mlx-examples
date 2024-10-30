import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

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
    use_cache: bool
    rms_norm: bool
    chunk_size: int
    tie_word_embeddings: bool
    intermediate_size: int = None
    time_step_limit: Tuple[float, float] = field(default_factory=lambda: (0.0, float("inf")))
    time_step_rank: Union[int, str] = "auto"
    model_type: str = "mamba2"

    def __post_init__(self):
        self.intermediate_size = int(self.expand * self.hidden_size) # E*D = ED

        if not hasattr(self, "head_dim"):
            self.head_dim = self.hidden_size // self.num_heads
        if self.time_step_rank == "auto":
            self.time_step_rank = math.ceil(self.hidden_size / 16)


class MambaRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = mx.ones(hidden_size)
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(mx.float32)

        if gate is not None:
            hidden_states = hidden_states * nn.functional.silu(gate.to(mx.float32))
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * math.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    

class Mamba2Mixer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Model dimensions
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_heads
        self.head_dim = args.head_dim
        self.ssm_state_size = args.state_size
        self.n_groups = args.n_groups
        self.intermediate_size = int(args.expand * args.hidden_size)
        
        # Convolution parameters
        self.conv_kernel = args.conv_kernel
        self.use_conv_bias = args.use_conv_bias
        
        # Time step parameters
        self.time_step_rank = int(args.time_step_rank)
        self.time_step_min = args.time_step_min
        self.time_step_max = args.time_step_max
        
        # Processing parameters
        self.chunk_size = args.chunk_size
        self.layer_norm_epsilon = args.layer_norm_epsilon
        
        # Calculate dimensions
        self.conv_dim = (self.intermediate_size + 
                        2 * self.n_groups * self.ssm_state_size)
        projection_size = (self.intermediate_size + 
                         self.conv_dim + 
                         self.num_heads)
        
        # Initialize layers
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=args.use_bias
        )
        
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel,
            groups=self.conv_dim,
            padding=self.conv_kernel - 1,
            bias=self.use_conv_bias
        )

        # Initialize parameters
        self.dt_bias = mx.ones(self.num_heads)
        A = mx.arange(1, self.num_heads + 1)
        self.A_log = mx.log(A)
        self.D = mx.ones(self.num_heads)
        
        # Output layers
        self.norm = MambaRMSNormGated(
            self.intermediate_size,
            eps=self.layer_norm_epsilon
        )
        self.out_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=args.use_bias
        )

    def reshape_into_chunks(self, tensor, pad_size, chunk_size):
        if pad_size > 0:
            pad_shape = list(tensor.shape)
            pad_shape[1] = pad_size
            padding = mx.zeros(pad_shape, dtype=tensor.dtype)
            tensor = mx.concatenate([tensor, padding], axis=1)
        
        chunk_shape = list(tensor.shape)
        chunk_shape[1] = -1
        chunk_shape.insert(2, chunk_size)
        return tensor.reshape(chunk_shape)

    def segment_sum(self, x):
        return mx.cumsum(x, axis=-1)

    def process_single_token(self, hidden_states, B, C, dt, cache):
        batch_size = hidden_states.shape[0]
        
        # Process convolution state
        if cache is not None:
            conv_state = cache.conv_states
            # Roll the conv state and update the last position
            conv_state = mx.roll(conv_state, shift=-1, axis=-1)
            # Create new conv state with updated last position
            new_conv_state = mx.array(conv_state)
            new_conv_state = new_conv_state.at[:, :, -1].add(hidden_states)
            conv_state = new_conv_state
            
            # Compute convolution
            conv_out = mx.sum(conv_state * self.conv1d.weight[:, 0, :], axis=-1)
            if self.use_conv_bias:
                conv_out = conv_out + self.conv1d.bias
            
            # Apply SiLU activation
            conv_out = mx.sigmoid(conv_out) * conv_out
            
        else:
            # Initialize new cache
            conv_state = mx.zeros((batch_size, self.conv_dim, self.conv_kernel - 1))
            conv_out = self.conv1d(hidden_states)
            conv_out = mx.sigmoid(conv_out) * conv_out
        
        # Process SSM
        dt = mx.clip(
            nn.softplus(dt + self.dt_bias),
            self.time_step_min,
            self.time_step_max
        )
        
        A = -mx.exp(self.A_log)
        dA = mx.exp(dt * A[None, :])
        
        if cache is not None:
            ssm_state = cache.ssm_states
        else:
            ssm_state = mx.zeros(
                (batch_size, self.num_heads, self.head_dim, self.ssm_state_size)
            )
        
        # Compute SSM updates
        dBx = mx.einsum('bh,bhs,bhd->bhds', dt, B, hidden_states)
        next_state = ssm_state * dA[:, :, None, None] + dBx
        y = mx.einsum('bhds,bhs->bhd', next_state, C)
        
        # Add skip connection
        y = y + hidden_states * self.D[None, :, None]
        
        return y, conv_state, next_state

    def process_long_sequence(self, hidden_states, B, C, dt, ssm_state):
        batch_size, seq_len = hidden_states.shape[:2]
        pad_size = self.chunk_size - (seq_len % self.chunk_size)
        
        # Reshape into chunks
        x_chunks = self.reshape_into_chunks(hidden_states, pad_size, self.chunk_size)
        B_chunks = self.reshape_into_chunks(B, pad_size, self.chunk_size)
        C_chunks = self.reshape_into_chunks(C, pad_size, self.chunk_size)
        
        # Process time steps
        dt = nn.softplus(dt + self.dt_bias)
        dt = mx.clip(dt, self.time_step_min)
        
        # Prepare matrices
        A = -mx.exp(self.A_log)
        A = A * dt[:, None]
        
        # Process chunks
        A_chunks = self.reshape_into_chunks(
            mx.broadcast_to(A, (batch_size, seq_len + pad_size, self.num_heads)),
            pad_size,
            self.chunk_size
        )
        
        # Compute cumulative sums
        A_cumsum = mx.cumsum(A_chunks, axis=-1)
        L = mx.exp(self.segment_sum(A_chunks))
        
        # Process diagonal blocks
        G = mx.einsum('...lhn,...shn->...lsh', C_chunks, B_chunks)
        M = G * L[..., None, :]
        Y_diag = mx.einsum('...lsh,...sh->...lh', M, x_chunks)
        
        # Process off-diagonal blocks
        decay_states = mx.exp(A_cumsum[..., -1:] - A_cumsum)
        B_decay = B_chunks * decay_states[..., None]
        states = mx.einsum('...shn,...sh->...hn', B_decay, x_chunks)
        
        # Combine results
        y = Y_diag + states
        
        # Remove padding if necessary
        if pad_size > 0:
            y = y[:, :seq_len]
        
        return y, ssm_state

    def __call__(self, x: mx.array, cache: Optional[Mamba2Cache] = None) -> mx.array:
        batch_size, seq_len, _ = x.shape
        
        # Project input
        projected_states = self.in_proj(x.squeeze(1))
        
        # Calculate d_mlp based on projection size
        d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size - 2 *
                self.n_groups * self.ssm_state_size - self.num_heads) // 2
        
        # Split projections with corrected dimensions
        splits = [
            d_mlp,                      # z0
            d_mlp,                      # x0
            self.intermediate_size,     # gate
            self.conv_dim,             # hidden_states
            self.num_heads             # dt
        ]
        
        z0, x0, x1, gate, hidden_states, dt = projected_states.split(splits, axis=-1)
        
        # Split hidden states into components
        x_conv, BC = mx.split(hidden_states, [self.intermediate_size], axis=-1)
        B, C = mx.split(BC, [self.n_groups * self.ssm_state_size], axis=-1)
        
        # Process based on sequence length
        if seq_len > 1 and cache is None:
            y, next_state = self.process_long_sequence(
                x_conv, B, C, dt,
                mx.zeros((batch_size, self.num_heads, self.head_dim, self.ssm_state_size))
            )
        else:
            # Reshape for single token processing
            x_conv = x_conv.reshape(batch_size, -1, self.head_dim)
            B = B.reshape(batch_size, self.num_heads, -1)
            C = C.reshape(batch_size, self.num_heads, -1)
            y, conv_state, next_state = self.process_single_token(x_conv, B, C, dt, cache)
            
            if cache is not None:
                cache.update(conv_state, next_state)
        
        # Apply normalization and final projection
        y = self.norm(y) * gate
        return self.out_proj(y)
    

class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mixer = Mamba2Mixer(args)
        self.norm = nn.RMSNorm(args.hidden_size)

    def __call__(self, x: mx.array, cache: Optional[Mamba2Cache] = None) -> mx.array:
        return self.mixer(self.norm(x), cache) + x

class Mamba2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [ResidualBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        x = self.embeddings(x)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, layer_cache in zip(self.layers, cache):
            x = layer(x, layer_cache)

        return self.norm_f(x)

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.backbone = Mamba2Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        B, T = inputs.shape

        x = self.backbone(inputs, cache)

        if self.args.tie_word_embeddings:
            logits = self.backbone.embeddings.as_linear(x)
        else:
            logits = self.lm_head(x)

        return logits
    
    def make_cache(self, batch_size=1):
        return [
            Mamba2Cache(
                batch_size=batch_size,
                conv_dim=self.args.intermediate_size + 2 * self.args.n_groups * self.args.state_size,
                kernel_size=self.args.conv_kernel,
                num_heads=self.args.num_heads,
                head_dim=self.args.head_dim,
                state_size=self.args.state_size
            ) 
            for _ in range(len(self.backbone.layers))
        ]
    
    def sanitize(self, weights):
        for k, v in weights.items():
            if "conv1d.weight" in k and v.ndim == 3:
                weights[k] = v.moveaxis(2, 1)
        return weights
    
    @property
    def layers(self):
        return self.backbone.layers
