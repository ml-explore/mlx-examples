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

        
def selective_scan(x, A, B, C, chunk_size):
    """
    Selective scan implementation for training.

    Arguments
        x: (batch, seqlen, n_heads, d_head)
        A: (batch, seqlen, n_heads)
        B: (batch, seqlen, n_heads, d_state)
        C: (batch, seqlen, n_heads, d_state)

    Return
        y: (batch, seqlen, n_heads, d_head)
    """
    assert x.shape[1] % chunk_size == 0
    
    # Reshape into chunks
    def chunk_reshape(m):
        shape = list(m.shape)
        shape[1:2] = [shape[1] // chunk_size, chunk_size]
        return m.reshape(shape)
    
    x, A, B, C = map(chunk_reshape, (x, A, B, C))
    A = mx.transpose(A, [0, 3, 1, 2])
    
    # Compute cumulative sums
    A_cumsum = mx.cumsum(A, axis=-1)
    
    # Process chunks
    L = mx.exp(selective_cumsum(A))
    Y_diag = mx.einsum('bclhn,bcshn,bhcls,bcshp->bclhp', C, B, L, x)
    
    decay_states = mx.exp(A_cumsum[..., -1:] - A_cumsum)
    states = mx.einsum('bclhn,bhcl,bclhp->bchpn', B, decay_states, x)
    
    initial_states = mx.zeros_like(states[:, :1])
    states = mx.concatenate([initial_states, states], axis=1)
    decay_chunk = mx.exp(selective_cumsum(mx.pad(A_cumsum[..., -1], ((0,0), (0,0), (1,0)))))
    new_states = mx.einsum('bhzc,bchpn->bzhpn', decay_chunk, states)
    states = new_states[:, :-1]
    
    state_decay_out = mx.exp(A_cumsum)
    Y_off = mx.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)
    
    Y = (Y_diag + Y_off).reshape((-1, x.shape[1] * chunk_size, *Y_diag.shape[-2:]))
    return Y

def selective_cumsum(x: mx.array) -> mx.array:
    """Stable selective cumulative sum calculation."""
    T = x.shape[-1]
    x = mx.repeat(x[..., None], T, axis=-1)
    mask = mx.tril(mx.ones((T, T)), k=-1)
    x = x * mask
    x_cumsum = mx.cumsum(x, axis=-2)
    mask = mx.tril(mx.ones((T, T)), k=0)
    return mx.where(mask, x_cumsum, float('-inf'))


class Mamba2Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        # Project input to get various components [z, x, B, C, dt]
        projection_size = (2 * args.intermediate_size + 2 * args.n_groups * args.state_size + args.num_heads)
        self.in_proj = nn.Linear(
            args.hidden_size,
            projection_size, 
            bias=args.use_bias
        )

        # Convolution layer
        conv_dim = args.intermediate_size + 2 * args.n_groups * args.state_size
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.conv_kernel,
            groups=conv_dim,
            padding=args.conv_kernel - 1,
            bias=args.use_conv_bias
        )

        # SSM parameters
        self.dt_bias = mx.zeros(args.num_heads)
        self.A_log = mx.zeros(args.num_heads)
        self.D = mx.ones(args.num_heads)

        # Output projections
        self.norm = nn.RMSNorm(args.intermediate_size, eps=args.layer_norm_epsilon)
        self.out_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=args.use_bias)

    def __call__(self, u: mx.array, cache=None) -> mx.array:
    #     return self.forward_training(x) if x.shape[1] > 1 else self.forward_inference(x, cache)

    # def forward_training(self, u: mx.array) -> mx.array:
    #     # Reset cache during training
    #     self.cache = None
        
    #     # Input projection and splitting
    #     zxbcdt = self.in_proj(u)
    #     z, xBC, dt = mx.split(
    #         zxbcdt,
    #         [
    #             self.args.hidden_size,
    #             self.args.hidden_size + 2 * self.args.state_size
    #         ],
    #         axis=-1
    #     )

    #     # Time step processing
    #     dt = mx.clip(
    #         nn.softplus(dt + self.dt_bias),
    #         self.args.time_step_min,
    #         self.args.time_step_max
    #     )

    #     # Convolution processing
    #     xBC_t = mx.transpose(xBC, [0, 2, 1])
    #     conv_out = self.conv1d(xBC_t)
    #     xBC = mx.transpose(conv_out, [0, 2, 1])[:, :u.shape[1]]
    #     xBC = mx.sigmoid(xBC) * xBC  # SiLU

    #     # Split states
    #     x, B, C = mx.split(
    #         xBC,
    #         [self.args.hidden_size, self.args.state_size],
    #         axis=-1
    #     )

    #     # Reshape for selective scan
    #     x = x.reshape((-1, x.shape[1], self.args.num_heads, self.args.head_dim))
    #     A = -mx.exp(self.A_log)

    #     # Apply selective scan
    #     y = selective_scan(
    #         x * dt[..., None],
    #         A * dt,
    #         B[..., None, :],
    #         C[..., None, :],
    #         self.args.chunk_size
    #     )

    #     # Output processing
    #     y = y + x * self.D[None, None, :, None]
    #     y = y.reshape((-1, y.shape[1], self.args.hidden_size))
    #     y = self.norm(y, z)
    #     y = self.out_proj(y)
        
    #     return y

    # def forward_inference(self, u: mx.array, cache=None) -> mx.array:
        # """
        # u: (B, 1, D)
        # cache: (h_cache, conv_cache)
        # """
        # """Single token processing during inference."""
        # assert u.shape[1] == 1, "Inference mode expects single token"
        
        # batch_size = u.shape[0]
        # # Use provided cache or create new one
        # self.cache = cache if cache is not None else Mamba2Cache.get_cache(self.args, batch_size, None)
        
        # # Project input
        # zxbcdt = self.in_proj(u.squeeze(1)) # (B, 2D)
        # d_mlp = (zxbcdt.shape[-1] - 2 * self.args.hidden_size - 2 * self.args.n_groups * self.args.state_size - self.args.num_heads) // 2

        # # (1, 768) (1, 0) (1, 0) (1, 256) (1, 0) (1, 3328)
        # y0, z0, x0, z, xBC, dt = mx.split(
        #     zxbcdt,
        #     [
        #         d_mlp,
        #         d_mlp,
        #         self.args.hidden_size,
        #         self.args.hidden_size + 2 * self.args.n_groups * self.args.state_size,
        #         self.args.num_heads
        #     ],
        #     axis=-1
        # )

        # # Update convolution state and apply
        # conv_state = self.cache.update_conv_state(xBC)
        # xBC = mx.sum(conv_state[:, :, -1] * mx.transpose(self.conv1d.weight, [1, 0, 2]),  axis=-1) # (B, D) (4, 1792)

        # if self.args.use_conv_bias:
        #     xBC = xBC + self.conv1d.bias

        # xBC = mx.sigmoid(xBC) * xBC  # SiLU (4, 1792)

        # # Split states and ensure proper shapes
        # a0, x, B, C = mx.split(
        #     xBC, # (4, 1792)
        #     [
        #         self.args.hidden_size,
        #         self.args.n_groups * self.args.state_size,
        #         self.args.n_groups * self.args.state_size
        #     ],
        #     axis=-1
        # )
        
        # # SSM step with explicit shapes
        # A = -mx.exp(self.A_log) # (num_heads) (24,)
        # print(A.shape) # (24,)
        # print(dt.shape) # (1, 3328)
        # dA = mx.exp(dt * A[None, :])  # Shape: (batch_size, num_heads) <------- her eis the error
        
        # # Reshape x considering intermediate size
        # # x shape should be (batch_size * num_heads, head_dim)
        # x = mx.reshape(x, (batch_size, self.args.num_heads, -1))
        # assert x.shape[-1] == self.args.head_dim, f"Head dimension mismatch: {x.shape[-1]} vs {self.args.head_dim}"
        
        # B = mx.reshape(B, (batch_size, -1))  # Should be (batch_size, state_size)
        # C = mx.reshape(C, (batch_size, -1))  # Should be (batch_size, state_size)
        
        # # Compute dBx with explicit shapes
        # dBx = mx.einsum('bh,bs,bhd->bhds', dt, B, x)
        
        # ssm_state = self.cache.update_ssm_state(dA, dBx)
        
        # y = mx.einsum('bhds,bs->bhd', ssm_state, C)
        # y = y + x * self.D[None, :, None]
        # y = mx.reshape(y, (batch_size, self.args.hidden_size))
        
        # # Output processing
        # y = self.norm(y, z)

        # if d_mlp > 0:
        #     y = mx.cat([nn.silu(z0) * x0, y], axis=-1)

        # y = self.out_proj(y)

        # return mx.expand_dims(y, 1)

        assert u.shape[1] == 1, "Inference mode expects single token"
        
        batch_size = u.shape[0]
        # Use provided cache or create new one
        self.cache = cache if cache is not None else Mamba2Cache.get_cache(self.args, batch_size, None)
        
        # Project input
        zxbcdt = self.in_proj(u.squeeze(1))  # (B, projection_size)
        
        # Calculate splits based on model dimensions
        d_mlp = self.args.intermediate_size
        d_state = self.args.state_size * self.args.n_groups
        
        # Split the projection into its components
        splits = [
            d_mlp,  # y0
            d_mlp,  # z0
            self.args.hidden_size,  # x0
            self.args.hidden_size,  # z
            d_state * 2,  # xBC (includes both B and C)
            self.args.num_heads  # dt
        ]
        
        y0, z0, x0, z, xBC, dt = mx.split(zxbcdt, splits[:-1], axis=-1)
        
        # Update convolution state and apply
        conv_state = self.cache.update_conv_state(xBC)
        xBC = mx.sum(conv_state[:, :, -1] * mx.transpose(self.conv1d.weight, [1, 0, 2]), axis=-1)
        
        if self.args.use_conv_bias:
            xBC = xBC + self.conv1d.bias
        
        xBC = mx.sigmoid(xBC) * xBC  # SiLU
        
        # Split states and reshape
        x, BC = mx.split(xBC, [self.args.intermediate_size], axis=-1)
        B, C = mx.split(BC, [d_state], axis=-1)
        
        # Reshape for SSM computation
        x = mx.reshape(x, (batch_size, self.args.num_heads, -1))  # (B, H, head_dim)
        B = mx.reshape(B, (batch_size, self.args.num_heads, -1))  # (B, H, state_per_head)
        C = mx.reshape(C, (batch_size, self.args.num_heads, -1))  # (B, H, state_per_head)
        
        # Process dt to match expected shape
        dt = mx.reshape(dt, (batch_size, self.args.num_heads))  # (B, H)
        dt = mx.clip(
            nn.softplus(dt + self.dt_bias),
            self.args.time_step_min,
            self.args.time_step_max
        )
        
        # SSM step
        A = -mx.exp(self.A_log)  # (H,)
        dA = mx.exp(dt * A[None, :])  # (B, H)
        
        # Compute dBx
        dBx = mx.einsum('bh,bhs,bhd->bhds', dt, B, x)
        
        # Update SSM state and compute output
        ssm_state = self.cache.update_ssm_state(dA, dBx)
        y = mx.einsum('bhds,bhs->bhd', ssm_state, C)
        y = y + x * self.D[None, :, None]
        
        # Reshape output
        y = mx.reshape(y, (batch_size, self.args.hidden_size))
        
        # Final output processing
        y = self.norm(y, z)
        
        if d_mlp > 0:
            y = mx.concat([nn.silu(z0) * x0, y], axis=-1)
        
        y = self.out_proj(y)
        
        return mx.expand_dims(y, 1)  # (B, 1, D)


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mixer = Mamba2Block(args)
        self.norm = nn.RMSNorm(args.hidden_size)

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        # x : (B, L, D)
        return self.mixer(self.norm(x), cache) + x # (B, L, D)


class Mamba2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [ResidualBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        # x : (B, L)
        x = self.embeddings(x)
        # x : (B, L, D)
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
        # inputs : (B, L)
        B, T = inputs.shape

        x = self.backbone(inputs, cache)

        if self.args.tie_word_embeddings:
            logits = self.backbone.embeddings.as_linear(x)
        else:
            logits = self.lm_head(x)

        return logits
    
    def make_cache(self, batch_size=1):
        return [Mamba2Cache(
            batch_size=batch_size,
            hidden_size=self.args.hidden_size,
            state_size=self.args.state_size,
            conv_kernel=self.args.conv_kernel,
            num_heads=self.args.num_heads,
            head_dim=self.args.head_dim
        ) for _ in range(len(self.backbone.layers))]
    
    def sanitize(self, weights):
        for k, v in weights.items():
            if "conv1d.weight" in k and v.ndim == 3:
                weights[k] = v.moveaxis(2, 1)
        return weights