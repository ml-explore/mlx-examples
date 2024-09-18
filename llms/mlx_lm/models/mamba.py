from dataclasses import dataclass

import math

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, MambaCache


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
    pscan: bool = False
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
        if self.time_step_rank == "auto":
            self.time_step_rank = math.ceil(self.hidden_size / 16)

class DepthWiseConv1d(nn.Module):
    def __init__(self, channels, kernel_size, bias=True, padding=0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight = mx.random.normal((channels, 1, kernel_size))
        self.bias = mx.zeros((channels,)) if bias else None

    def __call__(self, x, conv_state=None):
        B, L, C = x.shape
        K = self.kernel_size
        
        if conv_state is None:
            conv_state = mx.zeros((B, K - 1, C))
        
        x = mx.concatenate([conv_state, x], axis=1)
        
        output = []
        for i in range(K):
            slice = x[:, i:i+L, :]
            output.append(slice * self.weight[:, 0, i])
        y = mx.sum(mx.stack(output), axis=0)
        
        if self.bias is not None:
            y = y + self.bias.reshape(1, 1, -1)
        
        new_conv_state = x[:, -K+1:, :]
        
        return y, new_conv_state


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


def pscan_f(A, X):
    Aa = A
    Xa = X

    B, D, L, _ = A.shape

    num_steps = int(math.log2(L))

    # up sweep
    for k in range(num_steps):
        T = 2 * (Xa.shape[2] // 2)

        Aa = Aa[:, :, :T].reshape(B, D, T//2, 2, -1)
        Xa = Xa[:, :, :T].reshape(B, D, T//2, 2, -1)

        Xa[:, :, :, 1] += Aa[:, :, :, 1] * Xa[:, :, :, 0]
        Aa[:, :, :, 1] *= Aa[:, :, :, 0]

        A[:, :, 2**(k+1)-1::2**(k+1)] = Aa[:, :, :, 1]
        X[:, :, 2**(k+1)-1::2**(k+1)] = Xa[:, :, :, 1]

        Aa = Aa[:, :, :, 1]
        Xa = Xa[:, :, :, 1]

    # down sweep
    for k in range(num_steps-1, -1, -1):
        Aa = A[:, :, 2**k-1::2**k]
        Xa = X[:, :, 2**k-1::2**k]

        step_len = Xa.shape[2]
        T = 2 * (step_len // 2)

        if T < step_len:
            last_val_aa = Aa[:, :, -1] * Aa[:, :, -2]
            last_val_xa = Xa[:, :, -1] + Aa[:, :, -1] * Xa[:, :, -2]

        Aa = Aa[:, :, :T].reshape(B, D, T//2, 2, -1)
        Xa = Xa[:, :, :T].reshape(B, D, T//2, 2, -1)

        Xa[:, :, 1:, 0] += Aa[:, :, 1:, 0] * Xa[:, :, :-1, 1]
        Aa[:, :, 1:, 0] *= Aa[:, :, :-1, 1]

        if T == step_len:
            A[:, :, 2**k-1::2**(k+1)] = Aa[:, :, :, 0]
            X[:, :, 2**k-1::2**(k+1)] = Xa[:, :, :, 0]
        else:
            A[:, :, 2**k-1::2**(k+1)] = mx.concatenate([Aa[:, :, :, 0], mx.array([last_val_aa]).reshape(B, D, 1, -1)], axis=2)
            X[:, :, 2**k-1::2**(k+1)] = mx.concatenate([Xa[:, :, :, 0], mx.array([last_val_xa]).reshape(B, D, 1, -1)], axis=2)


def pscan(A_in, X_in):
    """
    Applies the parallel scan operation, as defined above. Returns a new array.

    Args:
        A_in: mx.array =-> Shape(B, L, ED, N)
        X_in: mx.array -> Shape (B, L, ED, N)

    Returns:
        H: mx.array -> Shape (B, L, ED, N)
    """
    A = A_in[:].transpose(0, 2, 1, 3)
    X = X_in[:].transpose(0, 2, 1, 3)
    pscan_f(A, X)
    return X.transpose(0, 2, 1, 3)


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

    def ssm_step(self, x, ssm_state=None):
        A = -mx.exp(self.A_log)  # (ED, N)
        D = self.D  # (ED,)

        deltaBC = self.x_proj(x)  # (B, time_step_rank+2*N)
        delta, B, C = mx.split(deltaBC, indices_or_sections=[self.time_step_rank, self.time_step_rank+self.ssm_state_size], axis=-1)
        delta = nn.softplus(self.dt_proj(delta))  # (B, ED)

        deltaA = mx.exp(mx.expand_dims(delta, -1) * A)  # (B, ED, N)
        deltaB = mx.expand_dims(delta, -1) * mx.expand_dims(B, 1)  # (B, ED, N)

        BX = deltaB * mx.expand_dims(x, -1)  # (B, ED, N)

        if self.training:
            new_ssm_state = BX
        else:
            if ssm_state is None:
                ssm_state = mx.zeros((x.shape[0], self.intermediate_size, self.ssm_state_size))  # (B, ED, N)
            new_ssm_state = deltaA * ssm_state + BX  # (B, ED, N)

        y = (new_ssm_state @ mx.expand_dims(C, -1)).squeeze(2)  # (B, ED)
        y = y + D * x  # (B, ED)

        if self.training:
            return y
        else:
            return y, new_ssm_state
    

    def ssm(self, x):
        # x : (B, L, ED)
        # y : (B, L, ED)

        A = -mx.exp(self.A_log) # (ED, N)
        D = self.D

        deltaBC = self.x_proj(x) # (B, L, dt_rank+2*N)

        delta, B, C = mx.split(deltaBC, indices_or_sections=[self.time_step_rank, self.time_step_rank+self.ssm_state_size], axis=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta = nn.softplus(self.dt_proj(delta)) # (B, L, ED)

        if self.args.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y
    

    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)
        deltaA = mx.exp(mx.expand_dims(delta, -1) * A) # (B, L, ED, N)
        deltaB = mx.expand_dims(delta, -1) * mx.expand_dims(B, 2) # (B, L, ED, N)

        BX = deltaB * mx.expand_dims(x, -1) # (B, L, ED, N)
        
        hs = pscan(deltaA, BX)

        y = (hs @ mx.expand_dims(C, -1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)
        
        y = y + D * x

        return y # (B, L, ED)

    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)
        _, L, _ = x.shape

        deltaA = mx.exp(mx.expand_dims(delta, -1) * A) # (B, L, ED, N)
        deltaB = mx.expand_dims(delta, -1) * mx.expand_dims(B, 2) # (B, L, ED, N)

        BX = deltaB * mx.expand_dims(x, -1) # (B, L, ED, N)

        h = mx.zeros([x.shape[0], self.config.d_inner, self.config.d_state]) # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = mx.stack(hs, axis=1)

        y = (hs @ mx.expand_dims(C, -1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)
        
        y = y + D * x

        return y # (B, L, ED)
    
    
    def __call__(self, x, cache: MambaCache, layer_idx: int):
        B, T, D = x.shape

        outputs = []
        for t in range(T):
            xt = x[:, t, :]  # (B, D)
            xz = self.in_proj(xt)  # (B, 2*ED)
            x_t, z_t = xz.split(indices_or_sections=2, axis=1)  # (B, ED), (B, ED)

            if self.training:
                conv_out, _ = self.conv1d(mx.expand_dims(x_t, 1))
            else:
                conv_state = cache.state[0][layer_idx]
                conv_out, new_conv_state = self.conv1d(mx.expand_dims(x_t, 1), conv_state)
                cache.state[0][layer_idx] = new_conv_state

            x_t = conv_out.squeeze(1)  # (B, ED)
            x_t = nn.silu(x_t)

            if self.training:
                y_t = self.ssm_step(x_t)
            else:
                ssm_state = cache.state[1][layer_idx]
                y_t, new_ssm_state = self.ssm_step(x_t, ssm_state)
                cache.state[1][layer_idx] = new_ssm_state

            z_t = nn.silu(z_t)

            output_t = y_t * z_t
            output_t = self.out_proj(output_t)  # (B, D)
            outputs.append(output_t)

        output = mx.stack(outputs, axis=1)  # (B, T, D)
        return output
    
class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mixer = MambaBlock(args)
        self.norm = nn.RMSNorm(args.hidden_size)

    def __call__(self, x: mx.array, cache: MambaCache, layer_idx: int):
        if x.ndim == 2:
            x = mx.expand_dims(x, 1)  # Make it (B, 1, D)

        output = self.mixer(self.norm(x), cache, layer_idx)
        output = output + x
        return output

class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [ResidualBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm_f = nn.RMSNorm(args.hidden_size)

    def __call__(self, x: mx.array, cache: MambaCache):
        x = self.embeddings(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, cache, i)
        return self.norm_f(x)

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.backbone = Mamba(args)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache = None):
        if inputs.ndim == 1:
            inputs = mx.expand_dims(inputs, 0)
        
        B, T = inputs.shape
        
        if not self.training and cache is None:
            cache = self.make_cache(batch_size=B)
        
        x = self.backbone(inputs, cache)
        
        if self.args.tie_word_embeddings:
            logits = self.backbone.embeddings.as_linear(x)
        else:
            logits = self.lm_head(x)

        return logits

    def make_cache(self, batch_size: int = 1):
        return MambaCache(
            num_layers=self.args.num_hidden_layers,
            batch_size=batch_size,
            conv_state_size=(self.args.conv_kernel - 1, self.args.intermediate_size),
            ssm_state_size=(self.args.intermediate_size, self.args.state_size)
        )

    @property
    def layers(self):
        return self.backbone.layers
    
    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_hidden_layers

    @property
    def n_kv_heads(self):
        return self.args.num_hidden_layers