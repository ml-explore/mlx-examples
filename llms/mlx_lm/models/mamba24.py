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
    

def pad_tensor_by_size(input_tensor: mx.array, pad_size: int):
    """
    Padding x tensor with `pad_size` on the seq_len dim (dim=1)

    Assumes that we only have tensors of either size 4 or 3
    """
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)

    return mx.pad(input_tensor, pad_shape, mode="constant", value=0)


def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    """
    Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and
    simultaneously splitting it into chunk sequences.

    Assumes that we only have tensors of either size 4 or 3
    """
    # [bsz, seq_len, ...] -> [bsz, seq_len multiple of chunk_size, ...]
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)

    if len(input_tensor.shape) == 3:
        # [bsz, seq_len multiple of chunk_size, num_heads] -> [bsz, -1, chunk_size, num_heads]
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        # [bsz, seq_len multiple of chunk_size, num_heads, head_dim or state_size] -> [bsz, -1, chunk_size, num_heads, head_dim or state_size]
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
        )


def segment_sum(input_tensor):
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.size(-1)
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
    mask = mx.tril(mx.ones(chunk_size, chunk_size, device=input_tensor.device), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    # 3. compute actual cumsum
    tensor_segsum = mx.cumsum(input_tensor, dim=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    mask = mx.tril(mx.ones(chunk_size, chunk_size, device=input_tensor.device), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -mx.inf)
    return tensor_segsum


class Mamba2Block(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.args = args

        self.hidden_size = args.hidden_size
        self.num_heads = args.num_heads
        self.head_dim = args.head_dim
        self.state_size = args.state_size
        self.n_groups = args.n_groups
        self.conv_kernel = args.conv_kernel
        self.intermediate_size = int(args.expand * args.hidden_size)
        self.time_step_rank = int(args.time_step_rank)
        self.time_step_min = args.time_step_min
        self.time_step_max = args.time_step_max
        self.chunk_size = args.chunk_size

        
        # Convolution dimension includes both intermediate sizes
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=args.use_conv_bias,
            kernel_size=args.conv_kernel,
            groups=self.conv_dim,
            padding=args.conv_kernel - 1
        )

        # Compute input projection dimension
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(args.hidden_size, projection_size, bias=args.use_bias)

        self.dt_bias = mx.ones(self.num_heads)
        A = mx.arange(1, self.num_heads + 1)
        self.A_log = mx.log(A)
        self.D = mx.ones(self.num_heads)

        self.norm = MambaRMSNormGated(self.intermediate_size, eps=args.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=args.use_bias)

    def __call__(self, input_states: mx.array, cache):
        batch_size, seq_len, _ = input_states.shape

        # Gated MLP's linear projection 
        projected_states = self.in_proj(input_states)  # [1, 1, projection_size]
        
        d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size - 
                2 * self.n_groups * self.state_size - self.num_heads) // 2
        
        # Split projected states
        *_, gate, hidden_states, dt = projected_states.split(
            [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], 
            axis=-1
        )
        # hidden_states shape: [1, 1, conv_dim]

        # Get SSM state from cache
        ssm_state = cache.ssm_states[self.layer_idx]
        
        if cache.seqlen_offset > 0:
            # Handle cached generation case
            conv_state = cache.conv_states[self.layer_idx]  # [batch, conv_dim, conv_kernel]
            conv_state = mx.roll(conv_state, shifts=-1, axis=-1)
            
            # Handle batched generation - states are copied through
            # Properly reshape hidden_states for the conv_state update
            conv_state = conv_state.at[:, :, -1].set(hidden_states[:, 0, :])
            cache.conv_states[self.layer_idx] = conv_state
            
            # Compute convolution output
            hidden_states = mx.sum(conv_state * self.conv1d.weight[:, 0, :], axis=-1)
            if self.args.use_conv_bias:
                hidden_states += self.conv1d.bias
            hidden_states = nn.silu(hidden_states)[:, None, ...]  # [batch, 1, conv_dim] : decoding
        
        else:
            # Handle normal forward pass
            # Properly transpose while preserving the sequence dimension
            hidden_states = hidden_states.transpose(0, 2, 1)  # [1, conv_dim, 1]
            
            # Pad the convolution state
            padding_size = self.conv_kernel - 1
            conv_state = mx.pad(
                hidden_states,
                ((0, 0), (0, 0), (padding_size, 0))
            )
            
            # Store in cache
            cache.conv_states[self.layer_idx] = conv_state
            
            # Apply convolution with proper padding
            hidden_states = self.conv1d(hidden_states)  # [1, conv_dim, 1]
            hidden_states = hidden_states.transpose(0, 2, 1)  # [1, 1, conv_dim]
            hidden_states = nn.silu(hidden_states)

        # Split hidden states for SSM computation
        hidden_states, B, C = mx.split(
            hidden_states, 
            [self.intermediate_size, self.n_groups * self.state_size, self.n_groups * self.state_size], 
            axis=-1
        )
                    
        # Compute A matrix
        A = -mx.exp(self.A_log)

        if cache is not None and cache.seqlen_offset > 0:
            # Note: there is no need to pad parameter matrices here, as there is just one new token
            # for batched generation
            dt = dt[:, None, ...] if dt.ndim == 2 else dt[:, 0, :][:, None, ...]
            dt = dt.transpose(0, 2, 1).expand(batch_size, dt.shape[-1], self.head_dim)
            # [num_heads] -> [num_heads, head_dim]
            dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)

            dt = nn.softplus(dt + dt_bias)
            dt = mx.clamp(dt, self.time_step_min) #, self.time_step_max)
            A = A[..., None, None].expand(self.num_heads, self.head_dim, self.state_size)
            # [bsz, num_heads, head_dim, state_size]
            dA = mx.exp(dt[..., None] * A)

            # Discretize B
            # [bsz, n_groups * state_size] -> [bsz, n_groups, 1, state_size] ->
            # -> [bsz, n_groups, group to head repetition factor, state_size] -> [bsz, num_heads, state_size]
            B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
            B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
            B = B.reshape(batch_size, -1, B.shape[-1])
            # [bsz, num_heads, head_dim, state_size]
            dB = dt[..., None] * B[..., None, :]

            # Discretize x into dB
            # [bsz, intermediate_size] -> [bsz, num_heads, head_dim]
            hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
            dBx = dB * hidden_states[..., None]

            # State calculation
            cache.ssm_states[self.layer_idx].copy_(
                cache.ssm_states[self.layer_idx] * dA + dBx
            )
            # Subsequent output
            # [bsz, n_groups * state_size] -> [bsz, num_heads, state_size]
            C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
            C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
            C = C.reshape(batch_size, -1, C.shape[-1])
            # [bsz, num_heads, head_dim]

            ssm_states = cache.ssm_states[self.layer_idx]  # Shape: [b, h, d, n]
            # Reshape ssm_states to merge the first two dimensions
            ssm_states_reshaped = ssm_states.view(batch_size * self.num_heads, self.head_dim, self.state_size)  # Shape: [b*h, d, n]
            C_reshaped = C.view(batch_size * self.num_heads, self.state_size, 1)  # Shape: [b*h, n, 1]
            y = ssm_states_reshaped @ C_reshaped
            y = y.view(batch_size, self.num_heads, self.head_dim)

            # D skip connection
            # [num_heads] -> [num_heads, head_dim]
            D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
            y = (y + hidden_states * D)

            # [bsz, num_heads, head_dim] -> [bsz, 1, intermediate_size]
            y = y.reshape(batch_size, -1)[:, None, ...]
        else:
            # begin ssd naive implementation without einsums
            dt = nn.functional.softplus(dt + self.dt_bias)
            dt = mx.clamp(dt, self.time_step_min)
            hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim)
            B = B.reshape(batch_size, seq_len,  -1, self.state_size)
            C = C.reshape(batch_size, seq_len, -1, self.state_size)
            B = B.repeat(1, 1, self.num_heads // self.n_groups, 1)
            C = C.repeat(1, 1, self.num_heads // self.n_groups, 1)
            pad_size = self.chunk_size - (seq_len % self.chunk_size)

            D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

            # Discretize x and A
            hidden_states = hidden_states * dt[..., None]
            A = A * dt

            # Rearrange into blocks/chunks
            hidden_states, A, B, C = [reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, A, B, C)]


            # [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
            A = A.permute(0, 3, 1, 2)
            A_cumsum = mx.cumsum(A, dim=-1)

            # 1. Compute the output for each intra-chunk (diagonal blocks)
            # This is the analog of a causal mask
            L = mx.exp(segment_sum(A))

            # First, contraction of C and B to get G (attention-weights like)
            G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, : ,:]  # shape: (b, c, l, s, h, n)
            G = G_intermediate.sum(dim=-1)  # shape: (b, c, l, s, h)


            # Step 2: Compute M, equivalent to applying attention mask to weights
            M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
            M = M_intermediate.sum(dim=-1)

            # Step 3: Compute Y_diag (apply to values)
            Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(3)

            # (right term of low-rank factorization of off-diagonal blocks; B terms)

            decay_states = mx.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
            B_decay_contraction = B * decay_states.permute(0, 2, 3, 1)[..., None]
            # permute back B * decay states
            states = (B_decay_contraction.permute(0, 1, 3, 2, 4)[..., None] * hidden_states.permute(0, 1, 3, 2, 4)[..., None, :]).sum(dim=3).permute(0, 1, 2, 4, 3)
            if cache is not None and cache.seqlen_offset > 0:
                previous_states = cache.ssm_states[self.layer_idx][:, None, ...]
            else:
                previous_states = mx.zeros_like(states[:, :1])
            states = mx.concat([previous_states, states], dim=1)
            decay_chunk = mx.exp(segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))

            states_permuted = states.permute(0, 2, 1, 3, 4)
            result = (decay_chunk[..., None, None] * states_permuted[:, :, None, ...]).sum(dim=2)
            new_states = result.permute(0, 2, 1, 3, 4)
            states, ssm_state = new_states[:, :-1], new_states[:, -1]

            # Compute state -> output conversion per chunk
            # (left term of low-rank factorization of off-diagonal blocks; C terms)
            state_decay_out = mx.exp(A_cumsum)
            # compute Yoff
            C_times_states = (C[..., None, :] * states[:, :, None, ...])
            state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
            Y_off = (C_times_states.sum(-1) * state_decay_out_permuted[..., None])
            # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)

            y = Y_diag + Y_off
            # [bsz, -1, self.chunk_size, num_heads, head_dim] -> [bsz, (padded) seq_len, num_heads, head_dim]
            y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)

            y = y + D_residual
            # Cutting off padded chunks
            if pad_size > 0:
                y = y[:, :seq_len, :, :]
            y = y.reshape(batch_size, seq_len, -1)

            if ssm_state is not None and cache is not None:
                cache.ssm_states[self.layer_idx] = ssm_state

        scan_output = self.norm(y, gate)
        # end ssd naive

        # 4. Final linear projection
        return self.out_proj(scan_output)  # [batch, seq_len, hidden_size]


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.residual_in_fp32 = args.residual_in_fp32
        self.mixer = Mamba2Block(args, layer_idx)
        self.norm = nn.RMSNorm(args.hidden_size)

    def __call__(self, x: mx.array, cache):
        return self.mixer(self.norm(x), cache) + x


class Mamba2(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [ResidualBlock(args, idx) for idx in range(args.num_hidden_layers)]
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
        return [Mamba2Cache(
            batch_size,
            self.args.intermediate_size,
            self.args.conv_kernel,
            self.args.head_dim,
            self.args.num_heads,
            self.args.n_groups,
            self.args.state_size
        ) for _ in range(len(self.layers))]
    
    def sanitize(self, weights):
        for k, v in weights.items():
            if "conv1d.weight" in k and v.ndim == 3:
                weights[k] = v.moveaxis(2, 1)
        return weights

    @property
    def layers(self):
        return self.backbone.layers