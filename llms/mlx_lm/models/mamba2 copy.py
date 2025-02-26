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


def segsum(x):
    """Stable segment sum calculation.

    `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.
    """
    T = x.shape[-1]
    x = mx.expand_dims(x, -1)
    x = mx.repeat(x, T, axis=-1)
    mask = mx.tril(mx.ones((T, T), dtype=mx.bool_), k=-1)
    x = mx.where(mask, x, 0)
    x_segsum = mx.cumsum(x, axis=-2)
    mask = mx.tril(mx.ones((T, T), dtype=mx.bool_), k=0)
    x_segsum = mx.where(mask, x_segsum, -mx.inf)
    return x_segsum

def ssd(x, A, B, C, chunk_size, initial_states=None):
    """Structured State Space Duality (SSD) - the core of Mamba-2

    Arguments
        x: (batch, seqlen, n_heads, d_head)
        A: (batch, seqlen, n_heads)
        B: (batch, seqlen, n_heads, d_state)
        C: (batch, seqlen, n_heads, d_state)

    Return
        y: (batch, seqlen, n_heads, d_head)
        final_state: final state for inference
    """
    assert x.shape[1] % chunk_size == 0

    # Rearrange into chunks
    def rearrange_to_chunks(m):
        shape = list(m.shape)
        shape[1:2] = [shape[1] // chunk_size, chunk_size]
        return m.reshape(shape)
    
    x_chunked = rearrange_to_chunks(x)
    A_chunked = rearrange_to_chunks(A)
    B_chunked = rearrange_to_chunks(B)
    C_chunked = rearrange_to_chunks(C)
    
    # Transpose A for easier cumsum
    A_chunked = mx.transpose(A_chunked, (0, 3, 1, 2))  # b c l h -> b h c l
    A_cumsum = mx.cumsum(A_chunked, axis=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = mx.exp(segsum(A_chunked))
    Y_diag = mx.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C_chunked, B_chunked, L, x_chunked)

    # 2. Compute the state for each intra-chunk
    decay_states = mx.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = mx.einsum("bclhn,bhcl,bclhp->bchpn", B_chunked, decay_states, x_chunked)

    # 3. Compute the inter-chunk SSM recurrence
    if initial_states is None:
        initial_states = mx.zeros_like(states[:, :1])
    states = mx.concatenate([initial_states, states], axis=1)
    
    A_cumsum_last = A_cumsum[:, :, :, -1]
    A_cumsum_padded = mx.pad(A_cumsum_last, [(0, 0), (0, 0), (1, 0)])
    decay_chunk = mx.exp(segsum(A_cumsum_padded))
    new_states = mx.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    state_decay_out = mx.exp(A_cumsum)
    Y_off = mx.einsum("bclhn,bchpn,bhcl->bclhp", C_chunked, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms
    Y_combined = Y_diag + Y_off
    
    # Reshape back to original sequence shape
    batch, chunks, chunk_len, heads, head_dim = Y_combined.shape
    Y = Y_combined.reshape(batch, chunks * chunk_len, heads, head_dim)

    return Y, final_state

def silu(x):
    """Applies the Sigmoid Linear Unit (SiLU), element-wise."""
    return x * mx.sigmoid(x)


class Mamba2Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.d_model = args.hidden_size
        self.d_state = args.state_size
        self.d_conv = args.conv_kernel
        self.expand = args.expand
        self.d_inner = int(self.expand * self.d_model)
        self.n_groups = args.n_groups
        self.n_heads = args.num_heads
        self.d_head = self.d_inner // self.n_heads
        self.chunk_size = args.chunk_size
        
        d_in_proj = 2 * self.d_inner + 2 * self.n_groups * self.d_state + self.n_heads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=args.use_bias)

        self.dt_bias = mx.random.normal((self.n_heads,)) * args.initializer_range
        self.A_log = mx.random.normal((self.n_heads,)) * args.initializer_range
        self.D = mx.random.normal((self.n_heads,)) * args.initializer_range

        # Use standard Conv1d with groups for depthwise convolution
        conv_dim = self.d_inner + 2 * self.n_groups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=self.d_conv,
            groups=conv_dim,  # Makes it depthwise
            padding=self.d_conv-1,
            bias=args.use_conv_bias
        )
        
        self.norm = nn.RMSNorm(self.d_inner, eps=args.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=args.use_bias)
    
    def __call__(self, u, cache=None):
        """
        Arguments
            u: (batch, seqlen, d_model) input
            cache: Optional tuple of (conv_state, ssm_state) for inference

        Return (y, cache)
            y: (batch, seqlen, d_model) output
            cache: updated tuple of (conv_state, ssm_state) for inference
        """
        if cache is not None:
            return self.step(u, cache)
        
        # Initialize cache if needed
        if cache is None:
            cache = [None, None]  # Initialize with None values

        # Compute projections
        zxbcdt = self.in_proj(u)

        # Split projections
        d_inner = self.d_inner
        d_state = self.n_groups * self.d_state
        
        z, xBC, dt = mx.split(
            zxbcdt,
            [d_inner, d_inner + 2 * d_state],
            axis=-1
        )
        
        # Process dt with softplus
        dt = mx.softplus(dt + self.dt_bias)  # (batch, seqlen, n_heads)
        
        # Apply convolution to xBC
        xBC_transposed = mx.transpose(xBC, (0, 2, 1))  # (batch, d, seqlen)
        xBC_conv = self.conv1d(xBC_transposed)
        xBC_conv = mx.transpose(xBC_conv, (0, 2, 1))  # (batch, seqlen, d)
        xBC = silu(xBC_conv[:, :u.shape[1], :])  # Ensure we only keep seqlen elements
        
        # Split xBC into x, B, C
        x, B, C = mx.split(
            xBC, 
            [d_inner, d_inner + d_state],
            axis=-1
        )
        
        # Reshape x for heads
        batch, seqlen = x.shape[0], x.shape[1]
        x_reshaped = x.reshape(batch, seqlen, self.n_heads, self.d_head)
        
        # Reshape B and C for SSM
        B = B.reshape(batch, seqlen, 1, d_state)
        C = C.reshape(batch, seqlen, 1, d_state)
        
        # Apply SSM with SSD algorithm
        A = -mx.exp(self.A_log)  # (n_heads,)
        A_dt = A * dt  # (batch, seqlen, n_heads)
        
        y, ssm_state = ssd(
            x_reshaped * mx.expand_dims(dt, -1),  # Scale x by dt
            A_dt,
            B,
            C,
            self.chunk_size
        )
        
        # Apply D and reshape
        y = y + x_reshaped * mx.reshape(self.D, (1, 1, self.n_heads, 1))
        y = y.reshape(batch, seqlen, d_inner)
        
        # Apply norm and gating
        y = self.norm(y, z)
        
        # Final projection
        y = self.out_proj(y)
        
        # Create cache for inference
        if seqlen == 1 and cache is not None:
            conv_state = mx.zeros((batch, d_inner + 2 * d_state, self.d_conv))
            conv_state = mx.update_slice(conv_state, xBC.reshape(batch, -1, 1), (0, 0, self.d_conv - 1))
            cache[0] = conv_state
            cache[1] = ssm_state
            
        return y, cache
    
    def step(self, u, cache):
        """Take an inference step for the current input and cache
        
        Arguments
            u: (batch, seqlen, d_model) - can be multiple tokens
            cache: tuple of (conv_state, ssm_state)
                
        Return (y, cache)
            y: (batch, seqlen, d_model)
            cache: updated cache object
        """
        batch, seqlen = u.shape[0], u.shape[1]
        
        # Initialize cache if it's None
        if cache[0] is None or cache[1] is None:
            d_state = self.n_groups * self.d_state
            conv_dim = self.d_inner + 2 * d_state
            conv_state = mx.zeros((batch, conv_dim, self.d_conv))
            
            # Fix: use correct state size per head
            state_per_head = d_state // self.n_heads
            ssm_state = mx.zeros((batch, self.n_heads, self.d_head, state_per_head))
        else:
            conv_state, ssm_state = cache[0], cache[1]
        
        # Project input
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        
        # Split projections
        d_inner = self.d_inner
        d_state = self.n_groups * self.d_state
        
        z, xBC, dt = mx.split(
            zxbcdt,
            [d_inner, d_inner + 2 * d_state],
            axis=-1
        )
        
        # Process each token through the convolution sequentially
        outputs = []
        for i in range(seqlen):
            # Get current token's input
            xBC_i = xBC[:, i]  # (batch, d_inner + 2*d_state)
            dt_i = dt[:, i]    # (batch, dt_size)
            
            # Extract the head-specific dt values
            dt_size = dt_i.shape[-1]
            
            if dt_size % self.n_heads == 0:
                # Reshape dt_i to extract the head-specific values
                dt_reshaped = dt_i.reshape(batch, self.n_heads, dt_size // self.n_heads)
                # Take the first element for each head
                dt_heads = dt_reshaped[:, :, 0]
            else:
                # If we can't reshape, just take the first n_heads elements
                dt_heads = dt_i[:, :self.n_heads]
            
            
            # Process dt with softplus
            dt_heads = nn.softplus(dt_heads + self.dt_bias.reshape(1, -1))  # (batch, n_heads)
            
            # Update convolution state
            conv_state = mx.roll(conv_state, shift=-1, axis=-1)
            
            # Use slice_update instead of update_slice
            # Reshape xBC_i to match the expected shape for the update
            xBC_reshaped = xBC_i.reshape(batch, -1, 1)
            # Create start_indices for the update
            start_indices = mx.array([0, 0, self.d_conv - 1])
            # Update the conv_state
            conv_state = mx.slice_update(
                conv_state,
                xBC_reshaped,
                start_indices,
                axes=(0, 1, 2)
            )
        
            # Apply convolution step
            weight = self.conv1d.weight
            bias = self.conv1d.bias if self.args.use_conv_bias else None
            
            xBC_conv = mx.sum(conv_state * weight.reshape(1, -1, self.d_conv), axis=-1)
            if bias is not None:
                xBC_conv = xBC_conv + bias
            
            xBC_conv = silu(xBC_conv)
            
            # Split xBC
            x_i, B_i, C_i = mx.split(
                xBC_conv, 
                [d_inner, d_inner + d_state],
                axis=-1
            )
            
            # Apply SSM step
            A = -mx.exp(self.A_log)  # (n_heads,)
            dA = mx.exp(dt_heads * A)  # (batch, n_heads)
            
            # Reshape x for heads
            x_i = x_i.reshape(batch, self.n_heads, self.d_head)
            
            # Reshape B and C for SSM with correct dimensions
            state_per_head = d_state // self.n_heads
            B_i_reshaped = B_i.reshape(batch, self.n_heads, state_per_head)
            C_i_reshaped = C_i.reshape(batch, self.n_heads, state_per_head)
            
            # Calculate dBx with the correctly shaped B
            dBx = mx.einsum("bhn,bhp->bhpn", B_i_reshaped, x_i * mx.expand_dims(dt_heads, -1))
            
            # Update SSM state
            ssm_state = ssm_state * mx.reshape(dA, (batch, self.n_heads, 1, 1)) + dBx
            
            # Calculate output with the correctly shaped C
            y_i = mx.einsum("bhpn,bhn->bhp", ssm_state, C_i_reshaped)
            
            # Apply D and reshape
            y_i = y_i + x_i * mx.reshape(self.D, (1, self.n_heads, 1))
            
            # Reshape y
            y_i = y_i.reshape(batch, d_inner)
            
            # Apply norm and gating (SwiGLU-like activation)
            y_i = self.norm(y_i)  # Just normalize without gating
            y_i = y_i * nn.sigmoid(z[:, i])  # Apply gating separately
            
            # Final projection
            y_i = self.out_proj(y_i)
            
            outputs.append(y_i)
        
        # Stack outputs along sequence dimension
        y = mx.stack(outputs, axis=1)  # (batch, seqlen, d_model)
        
        # Update cache
        cache[0] = conv_state
        cache[1] = ssm_state
        
        return y


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

    def make_cache(self):
        return [MambaCache() for _ in range(len(self.layers))]

    @property
    def layers(self):
        return self.backbone.layers
    









########################################################





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


def segsum(x):
    """Stable segment sum calculation.

    `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.
    """
    T = x.shape[-1]
    x = mx.expand_dims(x, -1)
    x = mx.repeat(x, T, axis=-1)
    mask = mx.tril(mx.ones((T, T), dtype=mx.bool_), k=-1)
    x = mx.where(mask, x, 0)
    x_segsum = mx.cumsum(x, axis=-2)
    mask = mx.tril(mx.ones((T, T), dtype=mx.bool_), k=0)
    x_segsum = mx.where(mask, x_segsum, -mx.inf)
    return x_segsum

def ssd(x, A, B, C, chunk_size, initial_states=None):
    """Structured State Space Duality (SSD) - the core of Mamba-2

    Arguments
        x: (batch, seqlen, n_heads, d_head)
        A: (batch, seqlen, n_heads)
        B: (batch, seqlen, n_heads, d_state)
        C: (batch, seqlen, n_heads, d_state)

    Return
        y: (batch, seqlen, n_heads, d_head)
        final_state: final state for inference
    """
    assert x.shape[1] % chunk_size == 0

    # Rearrange into chunks
    def rearrange_to_chunks(m):
        shape = list(m.shape)
        shape[1:2] = [shape[1] // chunk_size, chunk_size]
        return m.reshape(shape)
    
    x_chunked = rearrange_to_chunks(x)
    A_chunked = rearrange_to_chunks(A)
    B_chunked = rearrange_to_chunks(B)
    C_chunked = rearrange_to_chunks(C)
    
    # Transpose A for easier cumsum
    A_chunked = mx.transpose(A_chunked, (0, 3, 1, 2))  # b c l h -> b h c l
    A_cumsum = mx.cumsum(A_chunked, axis=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = mx.exp(segsum(A_chunked))
    Y_diag = mx.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C_chunked, B_chunked, L, x_chunked)

    # 2. Compute the state for each intra-chunk
    decay_states = mx.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = mx.einsum("bclhn,bhcl,bclhp->bchpn", B_chunked, decay_states, x_chunked)

    # 3. Compute the inter-chunk SSM recurrence
    if initial_states is None:
        initial_states = mx.zeros_like(states[:, :1])
    states = mx.concatenate([initial_states, states], axis=1)
    
    A_cumsum_last = A_cumsum[:, :, :, -1]
    A_cumsum_padded = mx.pad(A_cumsum_last, [(0, 0), (0, 0), (1, 0)])
    decay_chunk = mx.exp(segsum(A_cumsum_padded))
    new_states = mx.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    state_decay_out = mx.exp(A_cumsum)
    Y_off = mx.einsum("bclhn,bchpn,bhcl->bclhp", C_chunked, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms
    Y_combined = Y_diag + Y_off
    
    # Reshape back to original sequence shape
    batch, chunks, chunk_len, heads, head_dim = Y_combined.shape
    Y = Y_combined.reshape(batch, chunks * chunk_len, heads, head_dim)

    return Y, final_state

def silu(x):
    """Applies the Sigmoid Linear Unit (SiLU), element-wise."""
    return x * mx.sigmoid(x)


class Mamba2Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.d_model = args.hidden_size
        self.d_state = args.state_size
        self.d_conv = args.conv_kernel
        self.expand = args.expand
        self.d_inner = int(self.expand * self.d_model)
        self.n_groups = args.n_groups
        self.n_heads = args.num_heads
        self.d_head = self.d_inner // self.n_heads
        self.chunk_size = args.chunk_size
        
        d_in_proj = 2 * self.d_inner + 2 * self.n_groups * self.d_state + self.n_heads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=args.use_bias)

        self.dt_bias = mx.random.normal((self.n_heads,)) * args.initializer_range
        self.A_log = mx.random.normal((self.n_heads,)) * args.initializer_range
        self.D = mx.random.normal((self.n_heads,)) * args.initializer_range

        # Use standard Conv1d with groups for depthwise convolution
        conv_dim = self.d_inner + 2 * self.n_groups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=self.d_conv,
            groups=conv_dim,
            padding=self.d_conv-1,
            bias=args.use_conv_bias
        )
        
        self.norm = nn.RMSNorm(self.d_inner, eps=args.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=args.use_bias)
    
    def __call__(self, u, cache=None):
        """
        Arguments
            u: (batch, seqlen, d_model) input
            cache: Optional tuple of (conv_state, ssm_state) for inference

        Return (y, cache)
            y: (batch, seqlen, d_model) output
            cache: updated tuple of (conv_state, ssm_state) for inference
        """
        if cache is not None:
            return self.step(u, cache)
        
        # Initialize cache if needed
        if cache is None:
            cache = [None, None]  # Initialize with None values

        # Compute projections
        zxbcdt = self.in_proj(u)

        # Split projections
        d_inner = self.d_inner
        d_state = self.n_groups * self.d_state
        
        z, xBC, dt = mx.split(
            zxbcdt,
            [d_inner, d_inner + 2 * d_state],
            axis=-1
        )
        
        # Process dt with softplus
        dt = mx.softplus(dt + self.dt_bias)  # (batch, seqlen, n_heads)
        
        # Apply convolution to xBC
        xBC_transposed = mx.transpose(xBC, (0, 2, 1))  # (batch, d, seqlen)
        xBC_conv = self.conv1d(xBC_transposed)
        xBC_conv = mx.transpose(xBC_conv, (0, 2, 1))  # (batch, seqlen, d)
        xBC = silu(xBC_conv[:, :u.shape[1], :])  # Ensure we only keep seqlen elements
        
        # Split xBC into x, B, C
        x, B, C = mx.split(
            xBC, 
            [d_inner, d_inner + d_state],
            axis=-1
        )
        
        # Reshape x for heads
        batch, seqlen = x.shape[0], x.shape[1]
        x_reshaped = x.reshape(batch, seqlen, self.n_heads, self.d_head)
        
        # Reshape B and C for SSM
        B = B.reshape(batch, seqlen, 1, d_state)
        C = C.reshape(batch, seqlen, 1, d_state)
        
        # Apply SSM with SSD algorithm
        A = -mx.exp(self.A_log)  # (n_heads,)
        A_dt = A * dt  # (batch, seqlen, n_heads)
        
        y, ssm_state = ssd(
            x_reshaped * mx.expand_dims(dt, -1),  # Scale x by dt
            A_dt,
            B,
            C,
            self.chunk_size
        )
        
        # Apply D and reshape
        y = y + x_reshaped * mx.reshape(self.D, (1, 1, self.n_heads, 1))
        y = y.reshape(batch, seqlen, d_inner)
        
        # Apply norm and gating
        y = self.norm(y, z)
        
        # Final projection
        y = self.out_proj(y)
        
        # Create cache for inference
        if seqlen == 1 and cache is not None:
            conv_state = mx.zeros((batch, d_inner + 2 * d_state, self.d_conv))
            conv_state = mx.update_slice(conv_state, xBC.reshape(batch, -1, 1), (0, 0, self.d_conv - 1))
            cache[0] = conv_state
            cache[1] = ssm_state
            
        return y
    
    def step(self, u, cache):
        """Take an inference step for the current input and cache
        
        Arguments
            u: (batch, seqlen, d_model) - can be multiple tokens
            cache: tuple of (conv_state, ssm_state)
                
        Return (y, cache)
            y: (batch, seqlen, d_model)
            cache: updated cache object
        """
        batch, seqlen = u.shape[0], u.shape[1]
        
        # Initialize cache if it's None
        if cache[0] is None or cache[1] is None:
            d_state = self.n_groups * self.d_state
            conv_dim = self.d_inner + 2 * d_state
            conv_state = mx.zeros((batch, conv_dim, self.d_conv))
            
            # Fix: use correct state size per head
            state_per_head = d_state // self.n_heads
            ssm_state = mx.zeros((batch, self.n_heads, self.d_head, state_per_head))
        else:
            conv_state, ssm_state = cache[0], cache[1]
        
        # Project input
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        
        # Split projections
        d_inner = self.d_inner
        d_state = self.n_groups * self.d_state
        
        z, xBC, dt = mx.split(
            zxbcdt,
            [d_inner, d_inner + 2 * d_state],
            axis=-1
        )
        
        # Process dt with softplus once for all tokens
        dt_heads = dt.reshape(batch, seqlen, -1)[:, :, :self.n_heads]
        dt_heads = nn.softplus(dt_heads + self.dt_bias.reshape(1, 1, -1))
        
        # Pre-compute dA for all tokens
        A = -mx.exp(self.A_log)  # (n_heads,)
        dA = mx.exp(dt_heads * A.reshape(1, 1, -1))  # (batch, seqlen, n_heads)
        
        # Get convolution weights
        weight = self.conv1d.weight  # shape: (out_channels, 1, kernel_size)
        bias = self.conv1d.bias if self.args.use_conv_bias else None
        
        # Process each token through the convolution sequentially
        outputs = []
        for i in range(seqlen):
            # Get current token's input
            xBC_i = xBC[:, i]  # (batch, d_inner + 2*d_state)
            
            # Update convolution state
            conv_state = mx.roll(conv_state, shift=-1, axis=-1)
            
            # Update the last column of conv_state
            conv_state = mx.slice_update(
                conv_state,
                xBC_i.reshape(batch, -1, 1),
                mx.array([0, 0, self.d_conv - 1]),
                axes=(0, 1, 2)
            )
        
            # Apply convolution step - manually handle the depthwise conv
            # For a depthwise conv, we need to process each channel separately
            # conv_state shape: (batch, channels, kernel_size)
            # weight shape: (channels, 1, kernel_size) for depthwise conv
            
            # Reshape weight to match conv_state for element-wise multiplication
            # and then sum along the kernel dimension
            weight_reshaped = weight.reshape(conv_state.shape[1], self.d_conv)
            xBC_conv = mx.sum(conv_state * weight_reshaped.reshape(1, -1, self.d_conv), axis=-1)
            
            if bias is not None:
                xBC_conv = xBC_conv + bias
            
            xBC_conv = silu(xBC_conv)
            
            # Split xBC
            x_i, BC_rest = mx.split(xBC_conv, [d_inner], axis=-1)
            B_i, C_i = mx.split(BC_rest, [d_state], axis=-1)
            
            # Reshape x for heads
            x_i = x_i.reshape(batch, self.n_heads, self.d_head)
            
            # Reshape B and C for SSM
            state_per_head = d_state // self.n_heads
            B_i_reshaped = B_i.reshape(batch, self.n_heads, state_per_head)
            C_i_reshaped = C_i.reshape(batch, self.n_heads, state_per_head)
            
            # Get current token's dt and dA
            dt_i = dt_heads[:, i]  # (batch, n_heads)
            dA_i = dA[:, i]  # (batch, n_heads)
            
            # Calculate dBx
            dBx = mx.einsum("bhn,bhp->bhpn", B_i_reshaped, x_i * mx.expand_dims(dt_i, -1))
            
            # Update SSM state
            ssm_state = ssm_state * mx.reshape(dA_i, (batch, self.n_heads, 1, 1)) + dBx
            
            # Calculate output with the correctly shaped C
            y_i = mx.einsum("bhpn,bhn->bhp", ssm_state, C_i_reshaped)
            
            # Apply D and reshape
            y_i = y_i + x_i * self.D.reshape(1, self.n_heads, 1)
            
            # Reshape y
            y_i = y_i.reshape(batch, d_inner)
            
            # Apply norm and gating
            y_i = self.norm(y_i) * nn.sigmoid(z[:, i])
            
            # Final projection
            y_i = self.out_proj(y_i)
            
            outputs.append(y_i)
        
        # Stack outputs along sequence dimension
        y = mx.stack(outputs, axis=1)
        
        # Update cache
        cache[0] = conv_state
        cache[1] = ssm_state
        
        return y


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

    def make_cache(self):
        return [MambaCache() for _ in range(len(self.layers))]

    @property
    def layers(self):
        return self.backbone.layers