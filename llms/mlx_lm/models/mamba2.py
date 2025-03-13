import math
from dataclasses import dataclass
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


def segsum(x):
    return mx.cumsum(x, axis=-1).reshape(*x.shape[:-1], 1, x.shape[-1])


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
    prev_state=None,
) -> Tuple[mx.array, mx.array]:
    b, l, h, dh = x.shape
    _, _, g, _ = B.shape

    # Process dt
    if dt_bias is not None:
        dt = dt + dt_bias.reshape(1, 1, -1)
    dt = nn.softplus(dt)
    dt = mx.clip(dt, a_min=dt_min, a_max=dt_max)

    # Reshape tensors
    B_reshaped = mx.swapaxes(mx.swapaxes(B, 1, 3), 1, 2)
    C_reshaped = mx.swapaxes(C, 1, 2)

    # Compute CB
    CB = C_reshaped @ B_reshaped
    CB = mx.repeat(CB, repeats=h // g, axis=1)

    # Compute decay terms
    dtA = dt * A.reshape(1, 1, -1)
    dtA = mx.swapaxes(dtA, 1, 2)
    decay = mx.exp(segsum(dtA))

    # Create attention matrix
    surrogate_attention_matrix = mx.tril(CB * decay, 0)

    # Apply attention
    dtx = dt.reshape(b, l, h, 1) * x
    y = surrogate_attention_matrix @ dtx.swapaxes(1, 2)
    y = mx.swapaxes(y, 1, 2)

    # Compute next state
    decay_last = decay[:, :, -1, :].reshape(b, h, l).swapaxes(1, 2).reshape(b, l, h, 1)
    B_for_state = mx.repeat(B_reshaped, h // g, axis=1).swapaxes(2, 3)
    dtxdecay = dtx * decay_last
    dtxdecay = dtxdecay.swapaxes(1, 2).swapaxes(2, 3)
    
    # Calculate new state contribution
    new_state_contribution = dtxdecay @ B_for_state
    
    # Initialize or update state
    if prev_state is not None:
        decayed_prev_state = prev_state * decay[:, :, -1, :].reshape(b, h, 1, 1)
        next_state = decayed_prev_state + new_state_contribution
    else:
        next_state = new_state_contribution

    # Add skip connection if D is provided
    if D is not None:
        y += x * D.reshape(1, 1, h, 1)

    # Reshape output
    y = y.reshape(b, l, h * dh)

    return y, next_state


def ssd(x, A, B, C, chunk_size, initial_states=None):
    """Structured State Space Duality (SSD) - the core of Mamba-2
    
    Arguments
    x: (batch, seqlen, n_heads, d_head)
    A: (batch, seqlen, n_heads)
    B: (batch, seqlen, n_heads, d_state)
    C: (batch, seqlen, n_heads, d_state)
    
    Return (y, final_state)
    y: (batch, seqlen, n_heads, d_head)
    final_state: final state for next inference step
    """
    # Verify sequence length is divisible by chunk_size
    b, seqlen, h, dh = x.shape
    assert seqlen % chunk_size == 0
    
    # Rearrange into chunks
    num_chunks = seqlen // chunk_size
    x_chunks = x.reshape(b, num_chunks, chunk_size, h, dh)
    A_chunks = A.reshape(b, num_chunks, chunk_size, h)
    B_chunks = B.reshape(b, num_chunks, chunk_size, -1, B.shape[-1])  # Account for groups
    C_chunks = C.reshape(b, num_chunks, chunk_size, -1, C.shape[-1])
    
    # Transpose A for correct cumsum operation
    A_chunks = mx.transpose(A_chunks, (0, 3, 1, 2))  # b h c l
    A_cumsum = mx.cumsum(A_chunks, axis=-1)
    
    # 1. Compute output for each intra-chunk (diagonal blocks)
    L = mx.exp(segsum(A_chunks))
    
    # Handle the dimensions for einsum
    # "bclhn, bcshn, bhcls, bcshp -> bclhp"
    C_expanded = mx.expand_dims(C_chunks, axis=3)  # b c l 1 h n
    B_expanded = mx.expand_dims(B_chunks, axis=2)  # b c 1 s h n
    L_reshaped = mx.transpose(L, (0, 2, 3, 1, 4))  # b h c l s -> b c l h s
    x_reshaped = mx.transpose(x_chunks, (0, 1, 2, 3, 4))  # b c l h p
    
    # Perform the computation using manual broadcasting and reductions
    # This is a manual implementation of the einsum from PyTorch
    BC = mx.matmul(mx.transpose(C_expanded, (0, 1, 2, 4, 3)), 
                  mx.transpose(B_expanded, (0, 1, 3, 4, 2)))  # b c l n n
    L_x = mx.matmul(mx.transpose(L_reshaped, (0, 1, 2, 4, 3)), 
                   mx.reshape(x_reshaped, (b, num_chunks, chunk_size, dh, 1)))  # b c l s 1
    Y_diag = mx.matmul(BC, L_x)  # b c l h dh
    Y_diag = mx.reshape(Y_diag, (b, num_chunks, chunk_size, h, dh))
    
    # 2. Compute state for each intra-chunk
    decay_states = mx.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    
    # Compute states using matrix multiplications (replacing einsum)
    # "bclhn, bhcl, bclhp -> bchpn"
    B_decay = mx.matmul(B_chunks, 
                       mx.reshape(decay_states, (b, h, num_chunks, chunk_size, 1)))
    states = mx.matmul(B_decay, 
                      mx.reshape(x_chunks, (b, num_chunks, chunk_size, h, dh, 1)))
    states = mx.reshape(states, (b, num_chunks, h, dh, -1))  # b c h p n
    
    # 3. Compute inter-chunk recurrence
    if initial_states is None:
        initial_states = mx.zeros((b, 1, h, dh, B.shape[-1]))
    
    states = mx.concatenate([initial_states, states], axis=1)
    
    # Create padded A_cumsum for decay calculation
    A_cumsum_last = A_cumsum[:, :, :, -1]
    padded_A_cumsum = mx.pad(A_cumsum_last, [(0, 0), (0, 0), (1, 0)])
    decay_chunk = mx.exp(segsum(padded_A_cumsum))
    
    # Compute new states (replacing einsum "bhzc, bchpn -> bzhpn")
    decay_chunk_expanded = mx.reshape(decay_chunk, (b, h, -1, num_chunks+1, 1, 1))
    states_expanded = mx.reshape(states, (b, 1, num_chunks+1, h, dh, -1))
    new_states = decay_chunk_expanded * states_expanded
    new_states = mx.sum(new_states, axis=2)
    
    states, final_state = new_states[:, :-1], new_states[:, -1]
    
    # 4. Compute state -> output conversion per chunk
    state_decay_out = mx.exp(A_cumsum)
    
    # Compute Y_off (replacing einsum "bclhn, bchpn, bhcl -> bclhp")
    state_decay_expanded = mx.reshape(state_decay_out, (b, h, num_chunks, chunk_size, 1))
    states_reshaped = mx.reshape(states, (b, num_chunks, h, dh, -1))
    C_states = mx.matmul(mx.transpose(C_chunks, (0, 1, 2, 4, 3)), 
                         mx.transpose(states_reshaped, (0, 1, 3, 2, 4)))
    Y_off = C_states * state_decay_expanded
    Y_off = mx.sum(Y_off, axis=-1)
    Y_off = mx.reshape(Y_off, (b, num_chunks, chunk_size, h, dh))
    
    # Add diagonal and off-diagonal contributions
    Y = Y_diag + Y_off
    Y = mx.reshape(Y, (b, seqlen, h, dh))
    
    return Y, final_state

def ssd_inference_step(x, A, B, C, prev_state=None):
    """Simple inference step for Mamba-2
    
    Works with:
    - x: (batch, seqlen, n_heads, d_head)
    - A: (n_heads,) - scalar values
    - B: (batch, seqlen, n_groups, d_state)
    - C: (batch, seqlen, n_groups, d_state)
    """
    # Extract dimensions
    b, seqlen, h, dh = x.shape
    _, _, g, d_state = B.shape
    
    # Compute decay factor
    dA = mx.exp(A)  # (n_heads,)
    
    # Output container
    outputs = []
    
    # Final state to return
    final_state = prev_state
    
    # For each position in the sequence
    for t in range(seqlen):
        # Get current values
        xt = x[:, t]  # (batch, n_heads, d_head)
        Bt = B[:, t]  # (batch, n_groups, d_state)
        Ct = C[:, t]  # (batch, n_groups, d_state)
        
        # Handle groups vs heads if they differ
        if g < h:
            repeat_factor = h // g
            Bt = mx.repeat(Bt, repeat_factor, axis=1)  # (batch, n_heads, d_state)
            Ct = mx.repeat(Ct, repeat_factor, axis=1)  # (batch, n_heads, d_state)
        
        # Reshape for matrix operations
        xt = mx.reshape(xt, (b, h, dh, 1))
        Bt = mx.reshape(Bt, (b, h, 1, d_state))
        
        # Compute B·x
        dBx = mx.matmul(xt, Bt)  # (batch, n_heads, d_head, d_state)
        
        # Update state
        if final_state is not None:
            dA_expanded = mx.reshape(dA, (1, h, 1, 1))
            new_state = final_state * dA_expanded + dBx
        else:
            new_state = dBx
        
        # Compute output
        Ct = mx.reshape(Ct, (b, h, d_state, 1))
        yt = mx.matmul(new_state, Ct)  # (batch, n_heads, d_head, 1)
        yt = mx.reshape(yt, (b, h, dh))
        
        # Add to outputs
        outputs.append(mx.expand_dims(yt, 1))
        
        # Update state for next position
        final_state = new_state
    
    # Combine all outputs
    y = mx.concatenate(outputs, axis=1)
    
    return y, final_state

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

        self.conv1d = DepthWiseConv1d(
            channels=self.d_inner + 2 * self.n_groups * self.d_state,
            kernel_size=self.d_conv,
            bias=args.use_conv_bias,
            padding=self.d_conv-1
        )
        
        self.norm = nn.RMSNorm(self.d_inner, eps=args.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=args.use_bias)

    def __call__(self, u: mx.array, cache=None):
        batch_size, seq_len, _ = u.shape
        if cache is None:
            cache = [None, None]
        else:
            conv_state, ssm_state = cache
        
        zxBCdt = self.in_proj(u)
        
        # Split the projection into components
        z, xBC, dt = mx.split(
            zxBCdt,
            [self.d_inner, 2*self.d_inner + 2*self.n_groups*self.d_state],
            axis=-1
        )
        
        # Apply convolution and gating
        xBC, conv_state = self.conv1d(xBC, conv_state)
        xBC = xBC * mx.sigmoid(xBC)
        xBC = xBC[:, :seq_len, :]
        
        # Split into the various components
        x, B, C = mx.split(
            xBC,
            [self.d_inner, self.d_inner + self.d_state*self.n_groups],
            axis=-1
        )
        
        # Reshape for SSM computation
        x = mx.reshape(x, (batch_size, seq_len, self.n_heads, self.d_head))
        B = mx.reshape(B, (batch_size, seq_len, self.n_groups, -1))
        C = mx.reshape(C, (batch_size, seq_len, self.n_groups, -1))
        
        # Process dt - similar to your ssd_forward_attn function
        dt = mx.reshape(dt, (batch_size, seq_len, self.n_heads))
        dt = dt + self.dt_bias.reshape(1, 1, -1)  # Apply bias
        dt = nn.softplus(dt)  # Ensure positive time steps
        dt = mx.clip(dt, self.args.time_step_min, self.args.time_step_max)
        
        # For inference, we use ssd_forward_attn which you already know works
        y, next_ssm_state = ssd_forward_attn(
            x=x,
            dt=dt,
            A=self.A_log,  # Use A_log directly, the function will process it
            B=B,
            C=C,
            D=self.D,
            dt_bias=None,  # We already applied dt_bias above
            dt_min=self.args.time_step_min,
            dt_max=self.args.time_step_max,
            prev_state=ssm_state
        )
        
        # Reshape output
        y = mx.reshape(y, (batch_size, seq_len, self.d_inner))
        
        # Apply normalization and gating
        if self.args.norm_before_gate:
            y = self.norm(y)
            y = y * nn.silu(z)
        else:
            y = y * nn.silu(z)
            y = self.norm(y)
        
        y = self.out_proj(y)
        
        cache[0] = conv_state
        cache[1] = next_ssm_state
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