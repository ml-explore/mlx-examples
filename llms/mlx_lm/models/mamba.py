from dataclasses import dataclass

import math

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    state_size: int
    num_hidden_layers: int
    layer_norm_epsilon: float
    expand: int
    conv_kernel: int
    use_bias: bool
    use_conv_bias: bool
    initializer_range: float
    time_step_rank: int
    time_step_scale: float
    time_step_min: float
    time_step_max: float
    time_step_init_scheme: str
    time_step_floor: float
    rescale_prenorm_residual: bool
    use_cache: bool
    use_mambapy: bool = False
    dt_rank: str = "auto"

    def __post_init__(self):
        if not hasattr(self, 'hidden_size') and hasattr(self, 'd_model'):
            self.hidden_size = self.d_model
        if not hasattr(self, 'intermediate_size') and hasattr(self, 'd_inner'):
            self.intermediate_size = self.d_inner
        if not hasattr(self, 'state_size') and hasattr(self, 'd_state'):
            self.state_size = self.d_state
        if not hasattr(self, 'time_step_min') and hasattr(self, 'dt_min'):
            self.time_step_min = self.dt_min
        if not hasattr(self, 'time_step_max') and hasattr(self, 'dt_max'):
            self.time_step_min = self.dt_max
        if not hasattr(self, 'time_step_floor') and hasattr(self, 'dt_init_floor'):
            self.time_step_min = self.dt_init_floor
        if not hasattr(self, 'num_hidden_layers') and hasattr(self, 'n_layer'):
            self.num_hidden_layers = self.n_layer
        if not hasattr(self, 'num_hidden_layers') and hasattr(self, 'n_layers'):
            self.num_hidden_layers = self.n_layers
        if not hasattr(self, 'conv_kernel') and hasattr(self, 'd_conv'):
            self.conv_kernel = self.d_conv
        if not hasattr(self, 'use_bias') and hasattr(self, 'bias'):
            self.use_bias = self.bias
        if not hasattr(self, 'use_conv_bias') and hasattr(self, 'conv_bias'):
            self.use_conv_bias = self.conv_bias

        self.intermediate_size = self.expand * self.hidden_size
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.hidden_size / 16)


def clamp(x, min=None, max=None):
    if min is not None:
        mask_lower = x < min
    if max is not None:
        mask_upper = x > max
    if min is not None:
        if max is not None:
            return mx.where(mask_upper, max, mx.where(mask_lower, min, x))
        return mx.where(mask_lower, min, x)
    return mx.where(mask_upper, max, x)
    

class Conv1d(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        bias: bool = True,
        padding: int = 0
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.use_bias = bias
        self.padding = padding

        # Change the weight initialization to match the expected shape
        self.weight = mx.zeros((kernel_size, 1, channels))
        if self.use_bias:
            self.bias = mx.zeros((channels,))
        else:
            self.bias = None

    def __call__(self, x, cache=None):
        # Use the weight directly without transposing
        w = self.weight
        if cache is not None:
            l = []
            # Pad the cache if needed
            if cache.shape[1] < self.kernel_size - 1:
                l.append(
                    mx.zeros(
                        (x.shape[0], self.kernel_size - 1 - cache.shape[1], self.channels), dtype=x.dtype
                    )
                )
            l.extend([cache, x])
            x = mx.concatenate(l, axis=1)
            y = mx.conv_general(x, w, padding=([0], [0]), groups=self.channels)
        else:
            y = mx.conv_general(x, w, padding=([self.padding], [0]), groups=self.channels)

        # The cache is always kernel_size - 1
        cache = x[:, max(x.shape[1] - self.kernel_size + 1, 0) :, :]
        
        if self.use_bias:
            y = y + self.bias

        return y, cache


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(args.hidden_size, 2 * args.intermediate_size, bias=args.use_bias)

        # short 1d conv over time
        self.conv1d = Conv1d(
            channels=args.intermediate_size,
            kernel_size=args.conv_kernel,
            bias=args.use_conv_bias,
            padding=args.conv_kernel-1
        )

        # projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(args.intermediate_size, args.dt_rank + 2 * args.state_size, bias=False)

        # projects Δ from dt_rank to intermediate_size
        self.dt_proj = nn.Linear(args.dt_rank, args.intermediate_size, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = args.dt_rank**-0.5 * args.state_size
 
        if args.time_step_init_scheme == "constant":
            self.dt_proj.weight = dt_init_std * mx.ones_like(self.dt_proj.weight)
        elif args.time_step_init_scheme == "random":
            self.dt_proj.weight = mx.random.uniform(-dt_init_std, dt_init_std, self.dt_proj.weight.shape)
        else:
            raise NotImplementedError
        
        # dt bias
        dt = clamp(mx.exp(
            mx.random.uniform(shape=[args.intermediate_size]) * (math.log(args.time_step_max) - math.log(args.time_step_min)) + math.log(args.time_step_min)
        ), min=args.time_step_floor)
        inv_dt = dt + mx.log1p(-mx.exp(-dt))
        self.dt_proj.bias = inv_dt

        # S4D real initialization
        A = mx.repeat(mx.arange(1., 16 + 1.).reshape([1, 16]), repeats=args.intermediate_size, axis=0)
        self.A_log = mx.log(A) # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.D = mx.ones([args.intermediate_size])

        # projects block output from ED back to D
        self.out_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=args.use_bias)

    def ssm(self, x, h):
        # x : (B, ED)
        # h : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)

        A = -mx.exp(self.A_log) # (ED, N) # todo : move out of step (timestep independent)
        D = self.D

        deltaBC = self.x_proj(x) # (B, dt_rank+2*N)

        delta, B, C = mx.split(deltaBC, indices_or_sections=[self.args.dt_rank, self.args.dt_rank+self.args.state_size], axis=-1) # (B, dt_rank), (B, N), (B, N)
        delta = nn.softplus(self.dt_proj(delta)) # (B, ED)

        deltaA = mx.exp(mx.expand_dims(delta, -1) * A) # (B, ED, N)
        deltaB = mx.expand_dims(delta, -1) * mx.expand_dims(B, 1) # (B, ED, N)

        BX = deltaB * mx.expand_dims(x, -1) # (B, ED, N)

        if h is None:
            h = mx.zeros([x.shape[0], self.args.hidden_size, self.args.state_size]) # (B, ED, N)

        h = deltaA * h + BX # (B, ED, N)

        y = (h @ mx.expand_dims(C, -1)).squeeze(2) # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x
        
        return y, h
    
    def __call__(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
        # h : (B, ED, N)
        # inputs : (B, conv_kernel-1, ED)
        
        # y : (B, D)
        # cache : (h, inputs)
        
        h, inputs = cache
        
        print("Input shape:", x.shape)
        xz = self.in_proj(x) # (B, 2*ED)
        xz = xz.reshape(x.shape[0], -1)  # Ensure shape is (B, 2*ED)
        print("After in_proj shape:", xz.shape)
        x, z = xz.split(indices_or_sections=2, axis=1) # (B, ED), (B, ED)

        # x branch
        x_cache = mx.expand_dims(x, 1)
        x = self.conv1d(mx.concatenate([inputs, x_cache], axis=1))[:, self.args.conv_kernel-1, :] # (B, ED)

        x = nn.silu(x)
        y, h = self.ssm_step(x, h)

        # z branch
        z = nn.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, D)

        # prepare cache for next call
        inputs = mx.concatenate([inputs[:, 1:, :], x_cache], axis=1) # (B, conv_kernel-1, ED)
        cache = (h, inputs)
        
        return output, cache

class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mixer = MambaBlock(args)
        self.norm = nn.RMSNorm(args.hidden_size)

    def __call__(self, inputs: mx.array, cache):
        # x : (B, D)
        # cache : (h, inputs)
        # h : (B, ED, N)
        # inputs: (B, conv_kernel-1, ED)

        # output : (B, D)
        # cache : (h, inputs)

        output, cache = self.mixer(self.norm(inputs), cache)
        output = output + inputs
        return output, cache

class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [ResidualBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm_f = nn.RMSNorm(args.hidden_size)

    def __call__(self, tokens: mx.array, caches):
        # tokens : (B, L)

        # logits : (B, L, vocab_size)

        x = self.embeddings(tokens)

        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer(x, caches[i])

        return x, caches


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.backbone = Mamba(args)

    def __call__(self, inputs: mx.array, cache=None):
        out, cache = self.backbone(inputs, cache)
        # out = self.backbone.embeddings.as_linear(out)
        return out, cache

    @property
    def layers(self):
        return self.backbone.layers
    
    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_hidden_layers

    @property
    def n_kv_heads(self):
        return self.args.num_hidden_layers
    
    def make_cache(self):
        return [(None, mx.zeros([1, self.args.conv_kernel-1, self.args.intermediate_size])) for _ in range(self.args.num_hidden_layers)]

    def sanitize(self, weights):
        for key, value in weights.items():
            if "mixer.conv1d.weight" in key:
                # Ensure the weight is in the shape (kernel_size, 1, channels)
                if value.shape != (self.args.conv_kernel, 1, self.args.intermediate_size):
                    weights[key] = value.reshape(self.args.conv_kernel, 1, self.args.intermediate_size)
            elif key == "backbone.embeddings.weight":
                # Ensure the embedding weight is in the shape (vocab_size, hidden_size)
                if value.shape != (self.args.vocab_size, self.args.hidden_size):
                    weights[key] = value.T
        return weights