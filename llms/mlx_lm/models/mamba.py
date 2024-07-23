from dataclasses import dataclass
from typing import Optional, Union

import math

import torch

# import tokenizer

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "mamba"
    dt_rank: Union[int, str]
    d_model: int
    d_inner: int
    vocab_size: int
    n_layer: int
    use_bias: bool
    use_conv_bias: bool
    conv_kernel: int
    state_size: int
    expand: int
    time_step_init_scheme: str
    time_step_max: float
    time_step_min: float
    time_step_floor: float
    pscan: bool = False
    tie_word_embeddings: bool = False
    num_hidden_layers: int = None
    hidden_size: int = None

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        if self.n_layer is None:
            self.n_layer = self.num_hidden_layers
        if self.d_model is None:
            self.d_model = self.hidden_size
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


def pscan_main(A, X):
    Aa = A
    Xa = X
    B, D, L, _ = A.shape
    num_steps = int(math.log2(L))

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
    A = A_in[:].transpose(0, 2, 1, 3)
    X = X_in[:].transpose(0, 2, 1, 3)
    pscan_main(A, X)
    return X.transpose(0, 2, 1, 3)


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


def unsqueeze(x, axis):
    assert axis <= len(x.shape)
    if axis >= 0:
        new_shape = x.shape[:axis] + tuple([1]) + x.shape[axis:]
    else:
        new_shape = x.shape + tuple([1])
    return x.reshape(new_shape)


class DepthWiseConv1d(nn.Module):
    def __init__(self, channels, kernel_size, bias, padding):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = padding
        self.conv1d = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, bias=True, padding=padding)
        indices = mx.arange(channels)
        mask = mx.zeros_like(self.conv1d.weight)
        mask[indices, :, indices] = 1
        self.conv1d.weight *= mask

    def __call__(self, x):
        return self.conv1d(x)


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.in_proj = nn.Linear(args.d_model, 2 * args.d_inner, bias=args.use_bias)
        self.conv1d = DepthWiseConv1d(channels=args.d_inner, kernel_size=args.conv_kernel, bias=args.use_conv_bias, padding=args.conv_kernel-1)
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + 2 * args.state_size, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        dt_init_std = args.dt_rank**-0.5 * args.state_size
        if args.time_step_init_scheme == "constant":
            self.dt_proj.weight = dt_init_std * mx.ones_like(self.dt_proj.weight)
        elif args.time_step_init_scheme == "random":
            self.dt_proj.weight = mx.random.uniform(-dt_init_std, dt_init_std, self.dt_proj.weight.shape)
        else:
            raise NotImplementedError

        dt = clamp(mx.exp(mx.random.uniform(shape=[args.d_inner]) * (math.log(args.time_step_max) - math.log(args.time_step_min)) + math.log(args.time_step_min)), min=args.time_step_floor)
        self.dt_proj.bias = dt + mx.log1p(-mx.exp(-dt))
        A = mx.repeat(mx.arange(1., 16 + 1.).reshape([1, 16]), repeats=args.d_inner, axis=0)
        self.A_log = mx.log(A)
        self.D = mx.ones([args.d_inner])

        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.use_bias)

    def ssm_step(self, x, h):
        A = -mx.exp(self.A_log)
        D = self.D
        deltaBC = self.x_proj(x)
        delta, B, C = mx.split(deltaBC, indices_or_sections=[self.args.dt_rank, self.args.dt_rank+self.args.state_size], axis=-1)
        delta = nn.softplus(self.dt_proj(delta))
        deltaA = mx.exp(unsqueeze(delta, -1) * A)
        deltaB = unsqueeze(delta, -1) * unsqueeze(B, 1)
        BX = deltaB * unsqueeze(x, -1)
        if h is None:
            h = mx.zeros([x.shape[0], self.args.d_inner, self.args.state_size])
        h = deltaA * h + BX
        y = (h @ unsqueeze(C, -1)).squeeze(2)
        y = y + D * x
        return y, h

    def ssm(self, x): # DONE
        A = -mx.exp(self.A_log)
        D = self.D
        delta, B, C = self.x_proj(x).split(indices_or_sections=[self.args.dt_rank, self.args.dt_rank+self.args.state_size], axis=-1)
        delta = nn.softplus(self.dt_proj(delta))
        if self.args.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)
        return y

    def selective_scan(self, x, delta, A, B, C, D): # DONE
        deltaA = mx.exp(unsqueeze(delta, -1) * A)
        deltaB = unsqueeze(delta, -1) * unsqueeze(B, 2)
        BX = deltaB * unsqueeze(x, -1)
        hs = pscan(deltaA, BX)
        y = (hs @ unsqueeze(C, -1)).squeeze(3)
        return y + D * x

    def selective_scan_seq(self, x, delta, A, B, C, D):
        _, L, _ = x.shape
        deltaA = mx.exp(unsqueeze(delta, -1) * A)
        deltaB = unsqueeze(delta, -1) * unsqueeze(B, 2)
        BX = deltaB * unsqueeze(x, -1)
        h = mx.zeros([x.shape[0], self.args.d_inner, self.args.state_size])
        hs = []
        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
        hs = mx.stack(hs, axis=1)
        y = (hs @ unsqueeze(C, -1)).squeeze(3)
        return y + D * x

    def step(self, x, cache): # Done
        h, inputs = cache
        x, z = self.in_proj(x).split(indices_or_sections=2, axis=1)
        x_cache = unsqueeze(x, 1)
        x = self.conv1d(mx.concatenate([inputs, x_cache], axis=1))[:, self.args.conv_kernel-1, :]
        y, h = self.ssm_step(nn.silu(x), h)
        output = y * nn.silu(z)
        output = self.out_proj(output)
        inputs = mx.concatenate([inputs[:, 1:, :], x_cache], axis=1)
        return output, (h, inputs)

    def ssm_step(self, x, h): # Done
        A = -mx.exp(self.A_log)
        D = self.D
        delta, B, C = self.x_proj(x).split(indices_or_sections=[self.args.dt_rank, self.args.dt_rank+self.args.state_size], axis=-1) # (B, dt_rank), (B, N), (B, N)
        delta = nn.softplus(self.dt_proj(delta))
        deltaB = unsqueeze(delta, -1) * unsqueeze(B, 1)
        BX = deltaB * unsqueeze(x, -1)
        if h is None:
            h = mx.zeros([x.shape[0], self.args.d_inner, self.args.d_state])
        h = deltaA * h + BX
        y = (h @ unsqueeze(C, -1)).squeeze(2)
        y = y + D * x
        return y, h

    def __call__(self, x): # DONE
        _, L, _ = x.shape
        x, z = self.in_proj(x).split(indices_or_sections=2, axis=2)
        x = self.conv1d(x)[:, :L, :]
        output = self.ssm(nn.silu(x)) * nn.silu(z)
        return self.out_proj(output)


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mixer = MambaBlock(args)
        self.norm = nn.RMSNorm(args.d_model)

    def __call__(self, inputs: mx.array, cache: Optional[mx.array] = None):
        output, cache = self.mixer.step(self.norm(inputs), cache)
        output = output + inputs
        return output, cache


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = [ResidualBlock(args) for _ in range(args.n_layer)]
        self.norm_f = nn.RMSNorm(args.d_model)

    def __call__(self, inputs: mx.array, cache=None):
        tokens = self.embedding(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            h, cache[i] = layer(tokens, cache[i])
        h = self.norm_f(h)
        return h, cache


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.backbone = Mamba(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        out, cache = self.backbone(inputs, cache)

        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        return out, cache

    def torch_to_mlx_depthwise_weights(self, torch_weights):
        torch_weights = torch_weights.transpose(2, 1)
        channels, kernel_size, _ = torch_weights.shape

        mlx_weights = torch.zeros(channels, kernel_size, channels)

        indices = torch.arange(channels)
        if torch_weights[:, :, 0].type() == 'torch.BFloat16Tensor':
            mlx_weights[indices, :, indices] = torch_weights[:, :, 0].float()
        else:
            mlx_weights[indices, :, indices] = torch_weights[:, :, 0]

        return mlx_weights

    def sanitize(self, torch_state_dict):
        new_state_dict = {}
        for key, value in torch_state_dict.items():
            if 'conv1d.weight' in key:
                value = self.torch_to_mlx_depthwise_weights(value)

            if 'conv1d' in key:
                key = key.replace('conv1d', 'conv1d.conv1d')

            if value.type() == 'torch.BFloat16Tensor':
                new_state_dict[key] = value.half().numpy()
            else:
                new_state_dict[key] = value.numpy()

        return new_state_dict

    @property
    def layers(self):
        return self.model.layers

    def generate(self, tokenizer=None, prompt: str="Hello", n_tokens_to_gen: int = 50, sample: bool = True, temperature: float = 1.0, top_k: int = None):
        self.eval()

        input_ids = mx.array(tokenizer(prompt, return_tensors='np').input_ids)

        caches = [(None, mx.zeros([1, self.args.conv_kernel-1, self.args.d_inner])) for _ in range(self.args.n_layer)]

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

        output = [tokenizer.decode(output.tolist()) for output in input_ids][0]

        self.train()

        return output

# model = Model(ModelArgs())
# print(model)

# logits = model.generate()
# print(logits)
