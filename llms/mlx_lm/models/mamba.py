from dataclasses import dataclass
from typing import Optional, Union

import math
import einsum

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, MambaCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int # d_model
    intermediate_size: int # d_inner
    state_size: int # d_state
    num_hidden_layers: int # n_layer
    layer_norm_epsilon: float
    expand: int
    conv_kernel: int # d_conv
    use_bias: bool # bias
    use_conv_bias: bool # conv_bias
    initializer_range: float
    time_step_rank: int
    time_step_scale: float
    time_step_min: float
    time_step_max: float
    time_step_init_scheme: str
    time_step_floor: float
    rescale_prenorm_residual: bool
    use_cache: bool
    use_mambapy: bool = False # pscan
    dt_rank: str = "auto"

    def __post_init__(self):
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

        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            kernel_size=self.conv_kernel_size,
            bias=self.use_conv_bias,
            padding=self.conv_kernel_size-1
        )

        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        A = mx.repeat(mx.arange(1., self.ssm_state_size + 1), "n -> d n", repeats=self.intermediate_size)
        self.A_log = mx.log(A)
        self.D = mx.ones([self.intermediate_size])

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=args.use_bias)

    def ssm(self, x):
        (d_in, n) = self.A_log.shape

        A = -mx.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, time_step_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(indices_or_sections=[self.time_step_rank, n, n], axis=-1)  # delta: (b, l, time_step_rank). B, C: (b, l, n)
        delta = nn.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)
        
        return y
    
    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        deltaA = mx.exp(einsum(delta, A, 'b l d_in, d_in n -> b d_in l n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b d_in l n')
        x = mx.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = mx.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y = y + u * D
        return y

    def __call__(self, x):
        (b, l, d) = x.shape
        x_copy = x
        x, res = self.in_proj(self.norm(x)).split(indices_or_sections=[self.intermediate_size, self.intermediate_size], axis=-1)

        x = mx.rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = mx.rearrange(x, 'b d_in l -> b l d_in')
        
        x = nn.silu(x)

        y = self.ssm(x)
        
        y = y * nn.silu(res)
        return self.out_proj(y) + x_copy


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mixer = MambaBlock(args)
        self.norm = nn.RMSNorm(args.hidden_size)

    def __call__(self, inputs: mx.array):
        output = self.mixer(self.norm(inputs))
        output = output + inputs
        return output


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [ResidualBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm_f = nn.RMSNorm(args.hidden_size)

    def __call__(self, inputs: mx.array):
        tokens = self.embeddings(inputs)
        for i, layer in enumerate(self.layers):
            h, = layer(tokens)
        return self.norm_f(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.backbone = Mamba(args)
        # self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        out = self.backbone(inputs)
        out = self.backbone.embeddings.as_linear(out)
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
    
    def generate(self, input_ids: mx.array, n_tokens_to_gen: int = 50, sample: bool = True, temperature: float = 1.0, top_k: int = None):
        self.eval()

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        caches = [(None, mx.zeros([1, self.args.conv_kernel-1, self.args.intermediate_size])) for _ in range(self.args.num_hidden_layers)]

        for i in range(input_ids.shape[1] + n_tokens_to_gen - 1):
            next_token_logits, caches = self(input_ids[:, i], caches)

            if i+1 >= input_ids.shape[1]:

                if top_k is not None:
                    values = mx.topk(next_token_logits, k=top_k) # (1, k) ordered from lowest to biggest
                    mask = next_token_logits < (values[:, 0, None])
                    next_token_logits = mx.where(mask, -5000, next_token_logits) # TODO -mx.inf is problematic for now

                if sample and temperature > 0:
                    next_token = mx.random.categorical(next_token_logits * (1/temperature), num_samples=1)
                else:
                    next_token = mx.argmax(next_token_logits, axis=-1)[:, None]

                input_ids = mx.concatenate([input_ids, next_token], axis=1)

        self.train()
        return input_ids
