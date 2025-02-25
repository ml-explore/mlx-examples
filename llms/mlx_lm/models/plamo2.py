# Copyright © 2025 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs, create_attention_mask

from .cache import KVCache, MambaCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "plamo2"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    hidden_size_per_head: int = 128
    max_position_embeddings: int = 2048
    attention_window_size: int = 2048
    full_attention_idx: Optional[list[int]] = None
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_num_heads: int = 64
    mamba_step: int = 2
    mamba_chunk_size: int = 256
    mamba_enabled: bool = True
    intermediate_size: int = 13312
    vocab_size: int = 32000
    max_position_embeddings: int = 10 * 1024 * 1024


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        offset: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight = mx.zeros(hidden_size)
        self.variance_epsilon = eps
        self.offset = offset

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return mx.fast.rms_norm(
            hidden_states, self.weight + self.offset, self.variance_epsilon
        )


def get_initial_dt_bias(num_heads: int) -> mx.array:
    dt_min = 0.001
    dt_max = 0.1
    dt = mx.exp(
        mx.random.uniform(shape=(num_heads,)) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    )
    dt = mx.clip(dt, a_min=1e-4, a_max=None)
    inv_dt = dt + mx.log(-mx.expm1(-dt))
    return inv_dt


def get_initial_A(num_heads: int) -> mx.array:
    A = mx.arange(1, num_heads + 1, dtype=mx.float32)
    return mx.log(A)


# From: https://github.com/state-spaces/mamba/blob/0cce0fa645f100f00620ddf2333c2b7712abfdec/mamba_ssm/ops/triton/selective_state_update.py#L219
def selective_state_update_ref(
    state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False
) -> tuple[mx.array, mx.array]:
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, dim) or (batch, nheads, dim)
        dt: (batch, dim) or (batch, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, dstate) or (batch, ngroups, dstate)
        C: (batch, dstate) or (batch, ngroups, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, dim) or (batch, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, dim) or (batch, nheads, dim)
    """
    has_heads = state.ndim > 3
    if state.ndim == 3:
        state = mx.expand_dims(state, 1)
    if x.ndim == 2:
        x = mx.expand_dims(x, 1)
    if dt.ndim == 2:
        dt = mx.expand_dims(dt, 1)
    if A.ndim == 2:
        A = mx.expand_dims(A, 0)
    if B.ndim == 2:
        B = mx.expand_dims(B, 1)
    if C.ndim == 2:
        C = mx.expand_dims(C, 1)
    if D is not None and D.ndim == 1:
        D = mx.expand_dims(D, 0)
    if z is not None and z.ndim == 2:
        z = mx.expand_dims(z, 1)
    if dt_bias is not None and dt_bias.ndim == 1:
        dt_bias = mx.expand_dims(dt_bias, 0)
    batch, nheads, dim, dstate = state.shape
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
        dt = dt + dt_bias
    dt = nn.softplus(dt) if dt_softplus else dt
    dA = mx.exp(mx.expand_dims(dt, axis=-1) * A)  # (batch, nheads, dim, dstate)
    B = mx.reshape(
        mx.repeat(mx.expand_dims(B, axis=2), nheads // ngroups, 2),
        (batch, nheads, dstate),
    )  # (batch, nheads, dstate)
    C = mx.reshape(
        mx.repeat(mx.expand_dims(C, axis=2), nheads // ngroups, 2),
        (batch, nheads, dstate),
    )  # (batch, nheads, dstate)
    dB = mx.expand_dims(dt, axis=-1) * mx.expand_dims(
        B, axis=-2
    )  # (batch, nheads, dim, dstate)
    state = state * dA + dB * mx.expand_dims(x, axis=-1)  # (batch, dim, dstate)
    out = mx.einsum("bhdn,bhn->bhd", state.astype(C.dtype), C)
    if D is not None:
        out += (x * D).astype(out.dtype)
    out = (out if z is None else out * nn.silu(z)).astype(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    return out, state


def ssd_update_state(
    ssm_state: mx.array,
    x: mx.array,
    dt: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    z: mx.array,
    dt_bias: mx.array,
    dt_softplus: bool,
) -> tuple[mx.array, mx.array]:
    assert ssm_state.dtype == mx.float32
    dtype = x.dtype

    hidden_size_per_head = x.shape[-1]
    d_state = B.shape[-1]
    A = mx.broadcast_to(
        A[:, None, None], (A.shape[0], hidden_size_per_head, d_state)
    ).astype(mx.float32)
    dt = mx.broadcast_to(
        dt[..., None], (dt.shape[0], dt.shape[1], hidden_size_per_head)
    )
    dt_bias = mx.broadcast_to(
        dt_bias[:, None], (dt_bias.shape[0], hidden_size_per_head)
    )
    D = mx.broadcast_to(D[:, None], (D.shape[0], hidden_size_per_head))
    out, ssm_state = selective_state_update_ref(
        ssm_state,
        x.astype(dtype),
        dt.astype(dtype),
        A.astype(mx.float32),
        B.astype(dtype),
        C.astype(dtype),
        D.astype(mx.float32),
        z.astype(dtype),
        dt_bias.astype(mx.float32),
        dt_softplus=dt_softplus,
    )
    return out[:, None], ssm_state


def ssd_chunk_scan_combined(
    x: mx.array,
    dt: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    z: mx.array,
    dt_bias: mx.array,
    dt_softplus: bool,
    ssm_state: mx.array,
) -> tuple[mx.array, mx.array]:
    assert ssm_state.dtype == mx.float32
    length = x.shape[1]
    ys = []
    for i in range(length):
        y, ssm_state = ssd_update_state(
            ssm_state,
            x[:, i],
            dt[:, i],
            A,
            B[:, i],
            C[:, i],
            D if D.ndim == 1 else D[:, i],
            z=z[:, i],
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
        )
        ys.append(y)
    return mx.concatenate(ys, axis=1), ssm_state


def causal_conv1d_update(conv_state, x, weight) -> tuple[mx.array, mx.array]:
    batch, seqlen, dim = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-2]
    x = mx.concatenate([conv_state, x], axis=-2)
    conv_state = x[:, -state_len:]
    out = mx.conv1d(
        x,
        weight,
        padding=0,
        groups=dim,
    )[:, -seqlen:]
    return nn.silu(out), conv_state


class Mamba(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.d_state = config.mamba_d_state
        self.d_conv = config.mamba_d_conv
        self.chunk_size = config.mamba_chunk_size
        self.num_heads = config.mamba_num_heads
        self.hidden_size_per_head = config.hidden_size_per_head

        self.intermediate_size = self.num_heads * self.hidden_size_per_head

        self.in_proj = nn.Linear(
            self.hidden_size, 2 * self.intermediate_size, bias=False
        )
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=False,
            kernel_size=self.d_conv,
            groups=self.intermediate_size,
            padding=0,
        )
        self.dt_dim = max(64, self.hidden_size // 16)
        self.bcdt_proj = nn.Linear(
            self.intermediate_size,
            self.dt_dim + 2 * self.d_state,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_dim, self.num_heads, bias=False)

        self.dt_bias = get_initial_dt_bias(self.num_heads)
        self.A_log = get_initial_A(self.num_heads)
        self.D = mx.ones(self.num_heads, dtype=mx.float32)

        self.dt_norm_weight = mx.ones(self.dt_dim)
        self.B_norm_weight = mx.ones(self.d_state)
        self.C_norm_weight = mx.ones(self.d_state)

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
    ):
        bsize, length, _ = hidden_states.shape

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
            ssm_state = cache[1]
        else:
            conv_state = mx.zeros(
                (bsize, self.d_conv - 1, self.intermediate_size),
                dtype=hidden_states.dtype,
            )
            ssm_state = mx.zeros(
                (bsize, self.num_heads, self.hidden_size_per_head, self.d_state),
                dtype=mx.float32,
            )

        zx = self.in_proj(hidden_states)
        zx = zx.reshape(bsize, length, self.num_heads, -1)
        # z: (bsize, length, num_heads, hidden_size_per_head)
        # x: (bsize, length, num_heads, hidden_size_per_head)
        z, x = mx.split(
            zx,
            [
                self.hidden_size_per_head,
            ],
            axis=-1,
        )

        x = x.reshape(bsize, -1, self.num_heads * self.hidden_size_per_head)
        x, conv_state = causal_conv1d_update(conv_state, x, self.conv1d.weight)
        BCdt = self.bcdt_proj(x)
        x = x.reshape(bsize, length, self.num_heads, -1)
        B, C, dt = mx.split(BCdt, [self.d_state, self.d_state * 2], axis=-1)

        A = -mx.exp(self.A_log.astype(mx.float32))  # (num_heads,)
        dt = mx.fast.rms_norm(dt, self.dt_norm_weight, self.config.rms_norm_eps)
        B = mx.fast.rms_norm(B, self.B_norm_weight, self.config.rms_norm_eps)
        C = mx.fast.rms_norm(C, self.C_norm_weight, self.config.rms_norm_eps)

        # (bsize, length, num_heads, 1)
        dt = self.dt_proj(dt)[..., None]

        out, ssm_state = ssd_chunk_scan_combined(
            x,
            dt.reshape(bsize, length, -1),
            A,
            B,
            C,
            D=self.D,
            z=z,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            ssm_state=ssm_state,
        )

        if cache is not None:
            cache[0] = conv_state
            cache[1] = ssm_state
        y = self.out_proj(out.reshape(bsize, length, -1))

        return y


class Attention(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        head_dim = config.hidden_size_per_head
        self.max_position_embeddings = config.max_position_embeddings
        self.scale = head_dim**-0.5

        self.q_num_heads = config.num_attention_heads
        self.qk_dim = self.v_dim = head_dim
        self.k_num_heads = self.v_num_heads = config.num_key_value_heads
        assert self.q_num_heads % self.k_num_heads == 0
        self.n_group = self.q_num_heads // self.k_num_heads

        self.q_proj_dim = self.q_num_heads * self.qk_dim
        self.k_proj_dim = self.k_num_heads * self.qk_dim
        self.v_proj_dim = self.k_num_heads * self.v_dim
        self.qkv_proj = nn.Linear(
            self.hidden_size,
            self.q_proj_dim + self.k_proj_dim + self.v_proj_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.q_num_heads * self.v_dim, self.hidden_size, bias=False
        )

        self.q_weight = mx.ones((self.q_num_heads, self.qk_dim))
        self.k_weight = mx.ones((self.k_num_heads, self.qk_dim))

        self.rope = nn.RoPE(self.qk_dim)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
    ):
        B, T, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        q, k, v = mx.split(
            qkv, [self.q_proj_dim, self.q_proj_dim + self.k_proj_dim], axis=-1
        )
        q = q.reshape(B, T, self.q_num_heads, self.qk_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.k_num_heads, self.qk_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.v_num_heads, self.v_dim).transpose(0, 2, 1, 3)

        q = mx.fast.layer_norm(q, None, None, 1e-6) * self.q_weight[:, None]
        k = mx.fast.layer_norm(k, None, None, 1e-6) * self.k_weight[:, None]

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        output = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(
            B, T, self.q_num_heads * self.v_dim
        )
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size * 2, bias=False
        )
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.gate_up_proj(x)
        hs = mx.split(h, 2, axis=-1)
        return self.down_proj(nn.silu(hs[0]) * hs[1])


class PlamoDecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, is_mamba: bool) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.is_mamba = is_mamba
        self.mixer: nn.Module
        if is_mamba:
            self.mixer = Mamba(config)
        else:
            self.mixer = Attention(config)
        self.mlp = MLP(config)
        self.pre_mixer_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, offset=1.0
        )
        self.post_mixer_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, offset=1.0 / 5
        )
        self.pre_mlp_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, offset=1.0
        )
        self.post_mlp_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, offset=1.0 / (5**1.5)
        )

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
    ):
        residual = hidden_states
        hidden_states = self.pre_mixer_norm(hidden_states)

        hidden_states_sa = self.mixer(
            hidden_states=hidden_states,
            mask=mask,
            cache=cache,
        )

        hidden_states_sa = self.post_mixer_norm(hidden_states_sa)
        hidden_states = residual + hidden_states_sa

        residual = hidden_states
        hidden_states = self.pre_mlp_norm(hidden_states)

        # Fully Connected
        hidden_states_mlp = self.mlp(hidden_states)

        # Residual
        hidden_states_mlp = self.post_mlp_norm(hidden_states_mlp)
        return residual + hidden_states_mlp


def is_mamba(config: ModelArgs, i: int) -> bool:
    if not config.mamba_enabled:
        return False
    assert config.mamba_step > 1
    assert i < config.num_hidden_layers

    if config.num_hidden_layers <= (config.mamba_step // 2):
        # use attention in last layer
        return i != config.num_hidden_layers - 1
    return (i % config.mamba_step) != (config.mamba_step // 2)


class PlamoDecoder(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()

        self.layers = [
            PlamoDecoderLayer(config, is_mamba=is_mamba(config, i))
            for i in range(config.num_hidden_layers)
        ]

    def __call__(self, x: mx.array, mask: mx.array, cache):
        for i, decoder_layer in enumerate(self.layers):
            x = decoder_layer(
                x,
                mask=mask,
                cache=cache[i],
            )
        return x


class PlamoModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = PlamoDecoder(config)  # type: ignore
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
    ):
        batch_size, seq_length = inputs.shape

        h = self.embed_tokens(inputs)

        if mask is None:
            mask = create_attention_mask(h, [cache[1]] if cache is not None else None)

        if cache is None:
            cache = [None] * len(self.layers.layers)

        # decoder layers
        out = self.layers(
            h,
            mask,
            cache,
        )

        return self.norm(out)


class Model(nn.Module):

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = PlamoModel(config)

        self.vocab_size = config.vocab_size

        if not config.tie_word_embeddings:
            self.lm_head: nn.Module = nn.Linear(
                config.hidden_size, vocab_size, bias=False
            )

    def sanitize(self, weights: dict[Any, Any]) -> dict[Any, Any]:
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
        return weights

    def make_cache(self):
        # TODO use RotatingKVCache is not full_attn
        # full_attn = self.layer_idx in self.config.full_attention_idx
        return [MambaCache() if l.is_mamba else KVCache() for l in self.layers]

    def __call__(
        self, inputs: mx.array, mask: Optional[mx.array] = None, cache=None
    ) -> mx.array:
        outputs = self.model(
            inputs=inputs,
            mask=None,
            cache=cache,
        )
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(outputs)
        else:
            logits = self.lm_head(outputs)

        return logits

    @property
    def layers(self):
        return self.model.layers.layers
