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
    tie_word_embeddings: bool = True


    def __post_init__(self):
        if not hasattr(self, 'hidden_size') and hasattr(self, 'd_model'):
            self.hidden_size = self.d_model
        if not hasattr(self, 'intermediate_size') and hasattr(self, 'd_inner'):
            self.intermediate_size = self.d_inner
        if not hasattr(self, 'state_size') and hasattr(self, 'd_state'):
            self.state_size = self.d_state
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

class DepthWiseConv1d(nn.Module):
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
        self.padding = padding
        self.weight = mx.random.normal((channels, 1, kernel_size))
        if bias:
            self.bias = mx.zeros((channels,))
        else:
            self.bias = None

    def __call__(self, x, cache=None):
        B, L, C = x.shape
        assert C == self.channels, f"Input channels ({C}) must match the initialized channels ({self.channels})."
        
        w = self.weight  # Shape: (C, 1, K)
        K = self.kernel_size
        total_padding = self.padding + K - 1

        if cache is not None:
            l = []
            if cache.shape[1] < total_padding:
                l.append(mx.zeros((B, total_padding - cache.shape[1], C), dtype=x.dtype))
            l.extend([cache, x])
            x = mx.concatenate(l, axis=1)
        else:
            x = mx.pad(x, [(0, 0), (total_padding, 0), (0, 0)])

        # Manual depthwise convolution
        output = []
        for i in range(K):
            slice = x[:, i:i+L, :]
            output.append(slice * w[:, 0, i])
        y = mx.sum(mx.stack(output), axis=0)

        # The cache is always total_padding
        cache = x[:, max(x.shape[1] - total_padding, 0):, :]
        
        if self.bias is not None:
            y = y + self.bias.reshape(1, 1, -1)

        return y, cache


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


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.state_size
        self.conv_kernel_size = args.conv_kernel
        self.intermediate_size = args.intermediate_size
        self.time_step_rank = int(args.time_step_rank)
        self.use_conv_bias = args.use_conv_bias

        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=args.use_bias)

        self.conv1d = DepthWiseConv1d(
            channels=self.intermediate_size,
            kernel_size=self.conv_kernel_size,
            bias=self.use_conv_bias,
            padding=self.conv_kernel_size-1
        )

        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + 2 * self.ssm_state_size, bias=False)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        dt_init_std = args.time_step_rank**-0.5 * args.state_size
        if args.time_step_init_scheme == "constant":
            self.dt_proj.weight = dt_init_std * mx.ones_like(self.dt_proj.weight)
        elif args.time_step_init_scheme == "random":
            self.dt_proj.weight = mx.random.uniform(-dt_init_std, dt_init_std, self.dt_proj.weight.shape)
        else:
            raise NotImplementedError

        dt = clamp(mx.exp(
            mx.random.uniform(shape=[args.intermediate_size]) * (math.log(args.time_step_max) - math.log(args.time_step_min)) + math.log(args.time_step_min)
        ), min=args.time_step_floor)
        inv_dt = dt + mx.log1p(-mx.exp(-dt))
        self.dt_proj.bias = inv_dt

        A = mx.repeat(mx.arange(1., self.ssm_state_size + 1.).reshape([1, self.ssm_state_size]), repeats=self.intermediate_size, axis=0)
        self.A_log = mx.log(A)
        self.D = mx.ones([self.intermediate_size])

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=args.use_bias)

    def ssm_step(self, x, h):
        # x : (B, ED)
        # h : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)

        A = -mx.exp(self.A_log) # (ED, N) # todo : move out of step (timestep independent)
        D = self.D

        deltaBC = self.x_proj(x) # (B, dt_rank+2*N)

        delta, B, C = mx.split(deltaBC, indices_or_sections=[self.time_step_rank, self.time_step_rank+self.ssm_state_size], axis=-1) # (B, dt_rank), (B, N), (B, N)
        delta = nn.softplus(self.dt_proj(delta)) # (B, ED)

        deltaA = mx.exp(mx.expand_dims(delta, -1) * A) # (B, ED, N)
        deltaB = mx.expand_dims(delta, -1) * mx.expand_dims(B, 1) # (B, ED, N)

        BX = deltaB * mx.expand_dims(x, -1) # (B, ED, N)

        if h is None:
            h = mx.zeros([x.shape[0], self.intermediate_size, self.ssm_state_size]) # (B, ED, N)

        h = deltaA * h + BX # (B, ED, N)

        y = (h @ mx.expand_dims(C, -1)).squeeze(2) # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        return y, h
    
    def __call__(self, x, cache):
        # x : (B, T, D) where T is the number of tokens (5 in this case)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs : (B, d_conv-1, ED)

        h, inputs = cache
        B, T, D = x.shape

        outputs = []
        for t in range(T):
            xt = x[:, t, :]  # (B, D)
            xz = self.in_proj(xt)  # (B, 2*ED)
            x_t, z_t = xz.split(indices_or_sections=2, axis=1)  # (B, ED), (B, ED)

            # x branch
            x_cache = mx.expand_dims(x_t, 1)  # (B, 1, ED)
            conv_input = mx.concatenate([inputs, x_cache], axis=1)  # (B, d_conv, ED)
            conv_out, new_inputs = self.conv1d(conv_input)  # (B, d_conv, ED), (B, d_conv-1, ED)
            x_t = conv_out[:, -1, :]  # (B, ED)

            x_t = nn.silu(x_t)
            y_t, h = self.ssm_step(x_t, h)

            # z branch
            z_t = nn.silu(z_t)

            output_t = y_t * z_t
            output_t = self.out_proj(output_t)  # (B, D)
            outputs.append(output_t)

            # Update inputs for next token
            inputs = new_inputs

        output = mx.stack(outputs, axis=1)  # (B, T, D)
        cache = (h, inputs)

        return output, cache
    

class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mixer = MambaBlock(args)
        self.norm = nn.RMSNorm(args.hidden_size)

    def __call__(self, x: mx.array, cache):
        output, cache = self.mixer(self.norm(x), cache)
        output = output + x
        return output, cache


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [ResidualBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm_f = nn.RMSNorm(args.hidden_size)

    def __call__(self, x: mx.array, caches):
        x = self.embeddings(x)
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer(x, caches[i])
        return x, caches


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.backbone = Mamba(args)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        # inputs : (B, T) where T is the number of tokens
        # caches : [cache(layer) for all layers], cache : (h, inputs)
        
        if inputs.ndim == 1:
            inputs = mx.expand_dims(inputs, 0)  # Add batch dimension if not present
        
        B, T = inputs.shape
        x = self.backbone.embeddings(inputs)  # (B, T, D)
        
        for i, layer in enumerate(self.backbone.layers):
            x, cache[i] = layer(x, cache[i])
        
        x = self.backbone.norm_f(x)
        
        if self.args.tie_word_embeddings:
            logits = self.backbone.embeddings.as_linear(x)
        else:
            logits = self.lm_head(x)

        return logits, cache

    def make_cache(self):
        B = 1  # Assuming batch size of 1 for simplicity
        return [(None, mx.zeros((B, self.args.conv_kernel-1, self.args.intermediate_size))) 
                for _ in range(self.args.num_hidden_layers)]

    @property
    def layers(self):
        return self.backbone.layers
    
    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_hidden_layers

    @property
    def n_kv_heads(self):
        return self.args.num_hidden_layers
    
    # def make_cache(self):
    #     return [(None, mx.zeros([1, self.args.conv_kernel-1, self.args.intermediate_size])) for _ in range(self.args.num_hidden_layers)]