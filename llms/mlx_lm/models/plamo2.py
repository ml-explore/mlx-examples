import enum
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask


def _is_first_token(mask: mx.array) -> mx.array:
    assert mask.dtype == mx.bool_  # type: ignore
    B, Nh, q_len, kv_len = mask.shape
    mask = mask[:, :, :, -q_len:]
    cont = q_len != kv_len
    v = False if cont else True
    out = mx.logical_not(mx.diagonal(mask, offset=-1, axis1=-2, axis2=-1).astype(mx.bool_))  # type: ignore
    out = mx.concatenate([mx.full(shape=(B, Nh, 1), dtype=mx.bool_, vals=v), out], axis=-1)  # type: ignore
    return out


def _swiglu(h: mx.array) -> mx.array:
    size = h.shape[-1]
    chunks = 2
    _current_idx = 0
    split_sizes = []
    for i in range(chunks - 1):
        _current_idx += size // chunks + (1 if i < size % chunks else 0)
        split_sizes.append(_current_idx)
    hs = mx.split(h, split_sizes, axis=-1)
    return nn.silu(hs[0]) * hs[1]


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000) -> None:
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (mx.arange(0, self.dim, 2).astype(mx.float32) / self.dim))
        self._inv_freq = inv_freq

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(seq_len=max_position_embeddings, dtype=mx.float32)

    def _set_cos_sin_cache(self, seq_len: int, dtype: Any) -> None:
        self.max_seq_len_cached = seq_len
        t = mx.arange(self.max_seq_len_cached, dtype=self._inv_freq.dtype)  # type: ignore

        freqs = mx.einsum("i,j->ij", t, self._inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = mx.concatenate([freqs, freqs], axis=-1)
        self._cos_cached = emb.cos()[None, None, :, :].astype(mx.float32)
        self._sin_cached = emb.sin()[None, None, :, :].astype(mx.float32)

    def __call__(self, x: mx.array, seq_len: int) -> tuple[mx.array, mx.array]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self._cos_cached[:, :, :seq_len, ...].astype(x.dtype),  # type: ignore
            self._sin_cached[:, :, :seq_len, ...].astype(x.dtype),  # type: ignore
        )


def _rotate_half(x: mx.array) -> mx.array:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def _rotary_pos_emb(x: mx.array, cos: mx.array, sin: mx.array, position_ids: mx.array) -> mx.array:
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = mx.expand_dims(cos[position_ids], 1)  # [bs, 1, seq_len, dim]
    sin = mx.expand_dims(sin[position_ids], 1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (_rotate_half(x) * sin)
    return x_embed


class LinearType(str, enum.Enum):
    Normal = "normal"
    Fp8 = "fp8"
    Fp8Retain = "fp8-retain"


@dataclass
class ModelArgs(BaseModelArgs):  # type: ignore
    model_type: str = "plamo2"

    def __init__(
        self,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = True,
        # Attention
        num_attention_heads: int = 32,
        num_key_value_heads: int = 4,
        hidden_size_per_head: int = 128,
        max_position_embeddings: int = 2048,
        attention_window_size: int = 2048,
        full_attention_idx: list[int] | None = None,
        # Mamba
        mamba_d_state: int = 64,
        mamba_d_conv: int = 4,
        mamba_num_heads: int = 64,
        mamba_step: int = 2,
        mamba_chunk_size: int = 256,
        mamba_enabled: bool = True,
        # MLP
        intermediate_size: int = 13312,
        # Tokenizer
        vocab_size: int = 32000,
        tokenizer_class: str = "PlamoTokenizer",
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        # Multimodal
        image_token_id: Optional[int] = None,
        image_feature_size: Optional[int] = None,
        image_proj_type: Literal["linear", "mlp"] = "linear",
        # FP8
        linear_type: LinearType = LinearType.Normal,
        fp8_accum_dtype: Optional[str] = None,
        # Evaluation
        eval_attention_n_bit: Optional[int] = None,
        eval_mlp_n_bit: Optional[int] = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> None:
        # max_position_embeddings is often used to determine the max length during inference,
        # but samba should have extrapolation abilities
        self.max_position_embeddings = max(10 * 1024 * 1024, max_position_embeddings)
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps

        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size_per_head = hidden_size_per_head
        self.num_key_value_heads = num_key_value_heads
        self.attention_window_size = attention_window_size
        self.full_attention_idx = full_attention_idx if full_attention_idx is not None else []

        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_num_heads = mamba_num_heads
        self.mamba_step = mamba_step
        self.mamba_chunk_size = mamba_chunk_size
        self.mamba_enabled = mamba_enabled

        self.intermediate_size = intermediate_size

        self.vocab_size = vocab_size

        self.image_token_id = image_token_id
        self.image_feature_size = image_feature_size
        self.image_proj_type = image_proj_type

        self.linear_type = linear_type
        self.fp8_accum_dtype = fp8_accum_dtype

        self.eval_attention_n_bit = eval_attention_n_bit
        self.eval_mlp_n_bit = eval_mlp_n_bit
        self.use_cache = use_cache

        # fields for vLLM
        self.sliding_window = attention_window_size

        self.tokenizer_class = tokenizer_class
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

        # From PretrainedConfig of transformers
        self.use_return_dict = kwargs.pop("use_return_dict", True)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)


class PlamoAttentionCache(nn.Module):
    def __init__(self, key: mx.array, value: mx.array) -> None:
        super().__init__()
        B, nh, L, c = key.shape
        assert len(value.shape) == 4
        assert value.shape[0] == B
        assert value.shape[2] == L
        self.key = key
        self.value = value


class PlamoMambaCache(nn.Module):
    def __init__(self, conv_state: mx.array, ssm_state: mx.array) -> None:
        super().__init__()
        # conv_state: [B, C, d_conv]
        # ssm_state: [B, nhead, nchanel_per_head, d_state]
        assert len(conv_state.shape) == 3
        assert len(ssm_state.shape) == 4
        assert conv_state.shape[0] == ssm_state.shape[0]
        self.conv_state = conv_state
        self.ssm_state = ssm_state


PlamoLayerCache = PlamoAttentionCache | PlamoMambaCache


class PlamoCache(nn.Module):
    """
    stores states of the model for fast decoding.
    `transformers` uses `transformers.Cache` for this purpose, but the interface and variable names are
    deeply dependent on Transformers architecture (e.g., `key_states`) and it is difficult to use
    other architectures (e.g., Mamba).
    This class provides a similar interface to `transformers.Cache`, but is designed to also handle
    the state of Mamba properly.
    """

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.cache: list[Optional[PlamoLayerCache]] = [None for _ in range(config.num_hidden_layers)]

    def append_kv(self, key: mx.array, value: mx.array, layer_idx: int) -> tuple[mx.array, mx.array]:
        c = self.cache[layer_idx]
        if c is None:
            return key, value
        assert isinstance(c, PlamoAttentionCache)

        def _validate(cache: mx.array, new_tensor: mx.array) -> None:
            assert len(cache.shape) == 4
            assert len(new_tensor.shape) == 4
            assert cache.shape[0] == new_tensor.shape[0]
            assert cache.shape[1] == new_tensor.shape[1]
            assert cache.shape[3] == new_tensor.shape[3]

        _validate(c.key, key)
        _validate(c.value, value)
        assert key.shape[2] == value.shape[2]
        return mx.concatenate([c.key, key], axis=2), mx.concatenate([c.value, value], axis=2)

    def update_attention(self, key_states: mx.array, value_states: mx.array, layer_idx: int) -> PlamoAttentionCache:
        full_attn = layer_idx in self.config.full_attention_idx
        window_size = self.config.attention_window_size

        if self.cache[layer_idx] is None:
            if full_attn:
                self.cache[layer_idx] = PlamoAttentionCache(key_states, value_states)
            else:
                self.cache[layer_idx] = PlamoAttentionCache(
                    key_states[:, :, -window_size:, :],
                    value_states[:, :, -window_size:, :],
                )
        else:
            c = self.cache[layer_idx]
            assert isinstance(c, PlamoAttentionCache)
            k, v = self.append_kv(key_states, value_states, layer_idx)
            if full_attn:
                c.key = k
                c.value = v
            else:
                c.key = k[:, :, -window_size:, :]
                c.value = v[:, :, -window_size:, :]
            self.cache[layer_idx] = c
        return self.cache[layer_idx]  # type: ignore

    def update_mamba(self, conv_state: mx.array, ssm_state: mx.array, layer_idx: int) -> PlamoMambaCache:
        if self.cache[layer_idx] is None:
            self.cache[layer_idx] = PlamoMambaCache(conv_state, ssm_state)
        else:
            c = self.cache[layer_idx]
            assert isinstance(c, PlamoMambaCache)
            assert c.conv_state.shape == conv_state.shape
            assert c.ssm_state.shape == ssm_state.shape
            c.conv_state = conv_state
            c.ssm_state = ssm_state
        return self.cache[layer_idx]  # type: ignore

    def __getitem__(self, layer_idx: int) -> PlamoLayerCache | None:
        assert layer_idx < len(self.cache)
        layer_cache = self.cache[layer_idx]
        return layer_cache  # type: ignore
    
    @property
    def state(self):
        return self.cache

    @state.setter
    def state(self, v):
        self.cache = v

    def __len__(self) -> int:
        return len(self.cache)

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        if layer_idx is not None:
            c = self.cache[layer_idx]
            assert isinstance(c, PlamoAttentionCache)
            return c.key.shape[2]  # type: ignore

        sequence_length: int = 0
        for layer_cache in self.cache:
            if isinstance(layer_cache, PlamoAttentionCache):
                sequence_length = (
                    max(layer_cache.key.shape[2], sequence_length)
                    if sequence_length is not None
                    else layer_cache.key.shape[2]
                )
        return sequence_length

    def get_max_length(self) -> int | None:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def reorder_cache(self, beam_idx: mx.array) -> None:
        def _mamba(cache: PlamoMambaCache) -> PlamoMambaCache:
            return PlamoMambaCache(
                conv_state=mx.take(cache.conv_state, beam_idx, axis=0),
                ssm_state=mx.take(cache.ssm_state, beam_idx, axis=0),
            )

        def _attention(cache: PlamoAttentionCache) -> PlamoAttentionCache:
            return PlamoAttentionCache(
                key=mx.take(cache.key, beam_idx, axis=0),
                value=mx.take(cache.value, beam_idx, axis=0),
            )

        for i in range(len(self.cache)):
            if self.cache[i] is None:
                continue
            layer_cache = self.cache[i]
            if isinstance(layer_cache, PlamoMambaCache):
                self.cache[i] = _mamba(layer_cache)
            else:
                assert isinstance(layer_cache, PlamoAttentionCache)
                self.cache[i] = _attention(layer_cache)

    @property
    def seen_tokens(self) -> int | None:
        return None


class DecoderInput(NamedTuple):
    hidden_states: mx.array
    attention_mask: Optional[mx.array] = None
    past_states: Optional[PlamoCache] = None
    output_hidden_states: Optional[bool] = False
    output_attentions: Optional[bool] = False
    gradient_checkpointing: bool = False
    input_ids: Optional[mx.array] = None


class DecoderOutput(NamedTuple):
    hidden_states: mx.array
    all_hidden_states: Optional[tuple[mx.array, ...]]
    all_self_attns: Optional[tuple[mx.array, ...]]


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: tuple[int, int], dtype: mx.Dtype, past_key_values_length: int = 0) -> mx.array:
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = mx.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = mx.arange(mask.shape[-1])
    mask = mx.where(mask_cond < (mask_cond + 1).reshape((mask.shape[-1], 1)), 0, mask)
    mask = mask.astype(dtype)

    if past_key_values_length > 0:
        mask = mx.concatenate([mx.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], axis=-1)
    return mx.broadcast_to(mask[None, None, :, :], (bsz, 1, tgt_len, tgt_len + past_key_values_length))


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: mx.array, dtype: mx.Dtype, tgt_len: Optional[int] = None) -> mx.array:
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mx.broadcast_to(mask[:, None, None, :], (bsz, 1, tgt_len, src_len)).astype(dtype)

    inverted_mask = 1.0 - expanded_mask

    return mx.where(inverted_mask.astype(mx.bool_), float("-inf"), inverted_mask)  # type: ignore


def _rms_norm(hidden_states: mx.array, weight: Optional[mx.array], eps: float, offset: float = 1.0) -> mx.array:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.astype(mx.float32)
    variance = mx.power(hidden_states, 2).mean(-1, keepdims=True)
    hidden_states = hidden_states * mx.rsqrt(variance + eps)
    hidden_states = hidden_states.astype(input_dtype)
    if weight is not None:
        hidden_states = (offset + weight) * hidden_states
    return hidden_states


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
        return _rms_norm(hidden_states, self.weight, self.variance_epsilon, offset=self.offset)


def get_initial_dt_bias(num_heads: int) -> mx.array:
    dt_min = 0.001
    dt_max = 0.1
    dt = mx.exp(mx.random.uniform(shape=(num_heads,)) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
    dt = mx.clip(dt, a_min=1e-4, a_max=None)
    inv_dt = dt + mx.log(-mx.expm1(-dt))
    return inv_dt


def get_initial_A(num_heads: int) -> mx.array:
    A = mx.arange(1, num_heads + 1, dtype=mx.float32)
    return mx.log(A)


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
        mx.tile(mx.expand_dims(B, axis=2), (1, 1, nheads // ngroups, 1)),
        (batch, nheads, dstate),
    )  # (batch, nheads, dstate)
    C = mx.reshape(
        mx.tile(mx.expand_dims(C, axis=2), (1, 1, nheads // ngroups, 1)),
        (batch, nheads, dstate),
    )  # (batch, nheads, dstate)
    dB = mx.expand_dims(dt, axis=-1) * mx.expand_dims(B, axis=-2)  # (batch, nheads, dim, dstate)
    state = state * dA + dB * mx.expand_dims(x, axis=-1)  # (batch, dim, dstate
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

    hidden_size_per_head = x.shape[-1]
    d_state = B.shape[-1]
    A = mx.broadcast_to(A[:, None, None], (A.shape[0], hidden_size_per_head, d_state)).astype(mx.float32)
    dt = mx.broadcast_to(dt[..., None], (dt.shape[0], dt.shape[1], hidden_size_per_head))
    dt_bias = mx.broadcast_to(dt_bias[:, None], (dt_bias.shape[0], hidden_size_per_head))
    D = mx.broadcast_to(D[:, None], (D.shape[0], hidden_size_per_head))
    assert ssm_state.dtype == mx.float32
    out, ssm_state = selective_state_update_ref(
        ssm_state,
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        dt_bias,
        dt_softplus=dt_softplus,
    )
    return out[:, None], ssm_state


def _ssd_chunk_scan_combined_naive(
    x: mx.array,
    dt: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    z: mx.array,
    dt_bias: mx.array,
    dt_softplus: bool,
    seq_idx: mx.array | None,
    ssm_state: mx.array,
) -> tuple[mx.array, mx.array]:
    assert ssm_state.dtype == mx.float32
    length = x.shape[1]
    ys = []
    for i in range(length):
        if i != 0 and seq_idx is not None:
            ssm_state = mx.where(
                mx.array(seq_idx[:, i - 1] != seq_idx[:, i])[:, None, None, None],
                mx.zeros_like(ssm_state),
                ssm_state,
            )
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


def ssd_chunk_scan_combined(
    x: mx.array,
    dt: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    chunk_size: int,
    D: mx.array,
    z: mx.array,
    dt_bias: mx.array,
    dt_softplus: bool,
    seq_idx: mx.array | None,
    ssm_state: mx.array | None,
) -> tuple[mx.array, mx.array]:
    if seq_idx is not None:
        assert seq_idx.dtype == mx.int32
        assert ssm_state is None
    if ssm_state is not None:
        assert ssm_state.dtype == mx.float32
        assert seq_idx is None
    """
    state will be updates by following:
    ```
    dt = softplus(dt)
    dA = exp(dt * A)
    state_next = state * dA + dB * x
    ```
    To avoid updating state, we set dt to -inf and x to 0
    because `softplus(-inf) = 0` and `exp(0) = 1`
    """
    if ssm_state is None:
        bsize, _, num_heads, channel = x.shape
        state = B.shape[-1]
        ssm_state = mx.zeros((bsize, num_heads, channel, state), dtype=mx.float32)
    tmp, ssm_state = _ssd_chunk_scan_combined_naive(
        x,
        dt,
        A,
        B,
        C,
        D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        seq_idx=seq_idx,
        ssm_state=ssm_state,
    )
    return tmp, ssm_state


def _causal_conv1d(
    conv_state: mx.array | None, weight: mx.array, x: mx.array, seq_idx: mx.array | None
) -> tuple[mx.array, mx.array | None]:
    dtype = x.dtype
    if conv_state is not None:
        dtype = conv_state.dtype
        assert seq_idx is None
    if seq_idx is not None:
        assert seq_idx.dtype == mx.int32
        assert conv_state is None
    weight = weight.transpose(0, 2, 1).astype(dtype)
    x = x.astype(dtype)

    return_final_states = conv_state is not None
    if conv_state is None:
        bsize = x.shape[0]
        dim = weight.shape[0]
        d_conv = weight.shape[-1]
        conv_state = mx.zeros((bsize, dim, d_conv - 1), dtype=x.dtype)
    length = x.shape[-1]
    out = mx.zeros_like(x)
    for i in range(length):
        if i != 0 and seq_idx is not None:
            conv_state = mx.where(
                mx.array(seq_idx[:, i - 1] != seq_idx[:, i])[:, None, None],
                mx.zeros_like(conv_state),
                conv_state,
            )
        out[:, :, i : i + 1], conv_state = _causal_conv1d_update(conv_state, weight, x[:, :, i : i + 1])
    x = out
    if return_final_states:
        return x, conv_state
    else:
        return x, None


def causal_conv1d_update(
    x, conv_state, weight, bias=None, activation=None, cache_seqlens=None
) -> tuple[mx.array, mx.array]:
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.

    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.ndim == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = mx.concatenate([conv_state, x], axis=-1).astype(weight.dtype)  # (batch, dim, state_len + seqlen)
        conv_state = x_new[:, :, -state_len:]
    else:
        width_idx = mx.expand_dims(mx.arange(-(width - 1), 0, dtype=mx.int64), axis=0) + mx.expand_dims(
            cache_seqlens, axis=1
        )
        width_idx = mx.expand_dims(mx.remainder(width_idx, state_len), axis=1)
        width_idx = mx.broadcast_to(width_idx, (width_idx.shape[0], dim, width_idx.shape[2]))
        x_new = mx.concatenate([conv_state.gather(2, width_idx), x], axis=-1)
        x_new = x_new.astype(weight.dtype)
        copy_idx = mx.expand_dims(mx.arange(seqlen, dtype=mx.int64), axis=0) + mx.expand_dims(cache_seqlens, axis=1)
        copy_idx = mx.expand_dims(mx.remainder(copy_idx, state_len), axis=1)
        copy_idx = mx.broadcast_to(copy_idx, (copy_idx.shape[0], dim, copy_idx.shape[2]))
        conv_state.scatter_(2, copy_idx, x)
    assert bias is None
    # x_new: (N, C, L) -> (N, L, C)
    out = mx.conv1d(
        x_new.transpose(0, 2, 1),
        mx.expand_dims(weight, axis=2),
        padding=0,
        groups=dim,
    ).transpose(0, 2, 1)[:, :, -seqlen:]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else nn.silu(out)).astype(dtype_in), conv_state


def _causal_conv1d_update(conv_state: mx.array, weight: mx.array, xBC: mx.array) -> tuple[mx.array, mx.array]:
    dtype = conv_state.dtype
    xBC = xBC.astype(dtype)
    weight = weight.astype(dtype)

    x, conv_state = causal_conv1d_update(
        x=xBC,
        conv_state=conv_state,
        weight=weight[:, 0, :],
        activation="silu",
    )
    return x, conv_state


def is_mamba(config: ModelArgs, i: int) -> bool:
    if not config.mamba_enabled:
        return False
    assert config.mamba_step > 1
    assert i < config.num_hidden_layers

    if config.num_hidden_layers <= (config.mamba_step // 2):
        # use attention in last layer
        return i != config.num_hidden_layers - 1
    return (i % config.mamba_step) != (config.mamba_step // 2)


# Based on: https://github.com/Dao-AILab/causal-conv1d/blob/82867a9d2e6907cc0f637ac6aff318f696838548/causal_conv1d/causal_conv1d_interface.py#L206
def causal_conv1d(x, weight, bias=None, activation=None):
    """
    MLX implementation of a causal depthwise 1D convolution.
    Args:
        x (mx.array): Input tensor of shape (batch, channels, seq_len).
        weight (mx.array): Convolution filters of shape (channels, kernel_width).
                            Each channel has its own filter (depthwise conv).
        bias (mx.array, optional): Bias for each channel of shape (channels,).
        activation (str, optional): Activation to apply ("silu" or "swish" supported).
    Returns:
        mx.array: Output tensor of shape (batch, channels, seq_len).
    """
    x = mx.array(x) if not isinstance(x, mx.array) else x
    weight = mx.array(weight) if not isinstance(weight, mx.array) else weight
    if bias is not None:
        bias = mx.array(bias) if not isinstance(bias, mx.array) else bias

    batch, channels, seq_len = x.shape
    _, kernel_width = weight.shape  # weight shape: (channels, kernel_width)

    # Reshape weight for depthwise conv: (out_channels, in_channels/groups, kernel_width)
    # Here out_channels = channels, in_channels/groups = 1 (depthwise conv per channel)
    w = weight.reshape((channels, 1, kernel_width))

    # Pad input on the left with (kernel_width-1) zeros for causal convolution
    if kernel_width > 1:
        pad_shape = (batch, channels, kernel_width - 1)
        pad_zeros = mx.zeros(pad_shape, dtype=x.dtype)
        x_padded = mx.concatenate([pad_zeros, x], axis=2)  # concat along time axis
    else:
        x_padded = x

    # Perform depthwise convolution. Padding is already applied manually, so use padding=0 in conv1d.
    y = mx.conv1d(x_padded, w, stride=1, padding=0, groups=channels)
    # After convolution, y shape = (batch, channels, seq_len) because:
    # input length = seq_len + kernel_width - 1, no padding in conv, so output length = seq_len.

    # Add bias if provided (bias shape (channels,) broadcasts to (batch, channels, seq_len))
    if bias is not None:
        y = y + bias.reshape((1, channels, 1))

    # Apply activation if specified
    if activation in ("silu", "swish"):
        # SiLU (swish) activation: y * sigmoid(y)
        y = y * mx.sigmoid(y)
    elif activation is not None:
        raise ValueError(f"Unsupported activation: {activation}")

    return y


def swa_mask(q_len: int, kv_len: int, window_size: int) -> mx.array:
    max_len = max(q_len, kv_len)
    mask = mx.tril(
        mx.triu(mx.ones((max_len, max_len), dtype=mx.bool_), k=-window_size),  # type: ignore
        k=window_size,
    )
    return mask[-q_len:, -kv_len:]


class Attention(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
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
        self.o_proj = nn.Linear(self.q_num_heads * self.v_dim, self.hidden_size, bias=False)

        self.q_weight = mx.ones((self.q_num_heads, self.qk_dim))
        self.k_weight = mx.ones((self.k_num_heads, self.qk_dim))

        self.rotary_emb = RotaryEmbedding(self.qk_dim, max_position_embeddings=self.config.attention_window_size)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_states: Optional[PlamoCache] = None,
        output_attentions: bool = False,
    ) -> tuple[mx.array, Optional[mx.array], Optional[PlamoCache]]:
        bsz, q_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = mx.split(
            qkv, [self.q_proj_dim, self.q_proj_dim + self.k_proj_dim], axis=-1
        )
        query_states = query_states.reshape(bsz, q_len, self.q_num_heads, self.qk_dim).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(bsz, q_len, self.k_num_heads, self.qk_dim).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(bsz, q_len, self.v_num_heads, self.v_dim).transpose(0, 2, 1, 3)

        attn_dtype = query_states.dtype

        query_states = _rms_norm(query_states, None, 1e-6) * self.q_weight[None, :, None]
        key_states = _rms_norm(key_states, None, 1e-6) * self.k_weight[None, :, None]

        if past_states is not None:
            # reuse k, v, self_attention
            key_states_new = key_states
            value_states_new = value_states
            key_states, value_states = past_states.append_kv(key_states, value_states, self.layer_idx)  # type: ignore
            past_states.update_attention(key_states_new, value_states_new, self.layer_idx)

        kv_seq_len = key_states.shape[-2]
        position_ids = mx.arange(kv_seq_len, dtype=mx.int64)[None]
        q_position_ids = position_ids[:, -query_states.shape[2] :]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states = _rotary_pos_emb(query_states, cos, sin, q_position_ids)
        key_states = _rotary_pos_emb(key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        # expand shared kv
        assert self.k_num_heads == self.v_num_heads
        key_states = mx.tile(key_states, (1, self.n_group, 1, 1))
        value_states = mx.tile(value_states, (1, self.n_group, 1, 1))

        full_attn = self.layer_idx in self.config.full_attention_idx

        query_states = query_states.astype(attn_dtype)
        key_states = key_states.astype(attn_dtype)
        value_states = value_states.astype(attn_dtype)
        if attention_mask is not None and attention_mask.dtype != bool:
            attention_mask = attention_mask.astype(attn_dtype)
        if attention_mask is None:
            if not full_attn:
                assert key_states.shape[2] <= self.config.attention_window_size + 1
            mask = create_attention_mask(hidden_states)
            attn_output = mx.fast.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                scale=self.scale,
                mask=mask,
            )
        else:
            if attention_mask.dtype == bool:
                attention_mask = mx.where(attention_mask, mx.array(0.0, dtype=mx.float16), float("-inf"))
            if len(attention_mask.shape) == 2:
                attention_mask = attention_mask[None, None]
            assert len(attention_mask.shape) == 4

            if not full_attn:
                m_swa = swa_mask(
                    query_states.shape[2],
                    key_states.shape[2],
                    self.config.attention_window_size,
                )
                # `generate` function creates attention mask that does not consider sliding window
                m_swa = m_swa[None, None]
                attention_mask = attention_mask[:, :, -query_states.shape[2] :, -key_states.shape[2] :]
                attention_mask = mx.where(m_swa, attention_mask, float("-inf"))

            # like AttentionMaskConverter._unmask_unattended in huggingface.transfoermers,
            # we need to attend to all tokens in masked rows for `scaled_dot_product_attention`
            bool_mask = mx.logical_not(mx.isneginf(attention_mask))
            valid_tokens = mx.sum(bool_mask, axis=-1).astype(mx.bool_)  # type: ignore  # (..., q_len)
            attention_mask = mx.where(valid_tokens[..., None], attention_mask, float(0.0))
            attn_output = mx.fast.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                scale=self.scale,
                mask=attention_mask,
            )

        attn_output = attn_output.transpose(0, 2, 1, 3)

        attn_output = attn_output.reshape(bsz, q_len, self.q_num_heads * self.v_dim)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_states


class Mamba(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.d_state = config.mamba_d_state
        self.d_conv = config.mamba_d_conv
        self.chunk_size = config.mamba_chunk_size
        self.num_heads = config.mamba_num_heads
        # TODO add mamba_hidden_size_per_head config (?)
        self.hidden_size_per_head = config.hidden_size_per_head

        self.intermediate_size = self.num_heads * self.hidden_size_per_head

        self.in_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=False,  # TODO the original implementation uses bias
            kernel_size=self.d_conv,
            groups=self.intermediate_size,
            padding=0,
        )
        self.dt_dim = max(64, self.hidden_size // 16)
        # Notes:
        # Mamba2 removes this linear projection for simplicity (Figure 6 in the paper),
        # but it may degrade the ability of content-length extrapolation.
        self.bcdt_proj = nn.Linear(
            self.intermediate_size,
            self.dt_dim + 2 * self.d_state,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_dim, self.num_heads, bias=False)

        self.dt_bias = get_initial_dt_bias(self.num_heads)
        self.A_log = get_initial_A(self.num_heads)
        self.D = mx.ones(self.num_heads, dtype=mx.float32)

        # TODO norm weight before gating like Mamba2
        self.dt_norm_weight = mx.ones(self.dt_dim)
        self.B_norm_weight = mx.ones(self.d_state)
        self.C_norm_weight = mx.ones(self.d_state)

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def _no_weight_decay_param_names(self) -> set[str]:
        return set(["D", "dt_bias", "A_log"])

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_states: Optional[PlamoCache] = None,
    ) -> tuple[mx.array, Optional[PlamoCache]]:
        bsize, length, _ = hidden_states.shape
        is_update = length == 1 and past_states is not None

        bool_mask: mx.array | None = None
        seq_idx: mx.array | None = None
        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                attention_mask = mx.broadcast_to(
                    attention_mask[None, None],
                    (bsize, 1, attention_mask.shape[0], attention_mask.shape[1]),
                )
            assert len(attention_mask.shape) == 4

            if past_states is None:
                # TODO: support seq_idx with cache
                bool_mask_4d = mx.array(attention_mask == 0, dtype=mx.bool_)  # type: ignore
                is_first_token = _is_first_token(bool_mask_4d)[:, 0, :]
                seq_idx = mx.cumsum(is_first_token, axis=-1) - 1
                seq_idx = seq_idx.astype(mx.int32)

            # `generate` function creates attention mask that contains past tokens,
            # but mamba does not use them
            attention_mask = attention_mask[:, 0, -length:, -length:]
            bool_mask = mx.array(mx.diagonal(attention_mask, axis1=-2, axis2=-1) == 0)

        conv_state: mx.array | None
        ssm_state: mx.array | None
        if past_states is None:
            conv_state = None
            ssm_state = None
        elif past_states[self.layer_idx] is None:
            conv_state = mx.zeros(
                (bsize, self.intermediate_size, self.d_conv - 1),
                dtype=hidden_states.dtype,
            )
            ssm_state = mx.zeros(
                (bsize, self.num_heads, self.hidden_size_per_head, self.d_state),
                dtype=mx.float32,
            )
        else:
            c = past_states[self.layer_idx]
            assert isinstance(c, PlamoMambaCache)
            conv_state = c.conv_state
            ssm_state = c.ssm_state

        zx = self.in_proj(hidden_states)
        zx = zx.reshape(bsize, length, self.num_heads, -1).astype(mx.float32)
        # z: (bsize, length, num_heads, hidden_size_per_head)
        # x: (bsize, length, num_heads, hidden_size_per_head)
        z, x = mx.split(
            zx,
            [
                self.hidden_size_per_head,
            ],
            axis=-1,
        )

        # conv
        x = x.reshape(bsize, length, -1).transpose(0, 2, 1)  # (bsize, intermediate_size, length)
        if bool_mask is not None:
            x = mx.where(bool_mask[:, None, :], x, 0.0)
        if is_update:
            assert conv_state is not None
            x, conv_state = _causal_conv1d_update(conv_state, self.conv1d.weight, x)
        else:
            x, conv_state = _causal_conv1d(conv_state, self.conv1d.weight, x, seq_idx=seq_idx)
        x = x.astype(mx.float32)
        x = x.transpose(0, 2, 1)  # (bsize, length, intermediate_size)
        x = x.reshape(bsize, length, -1)
        # x: (bsize, length, num_heads, hidden_size_per_head)
        # B: (bsize, length, 1, d_state)
        # C: (bsize, length, 1, d_state)
        # dt: (bsize, length, dt_dim)
        BCdt = self.bcdt_proj(x).astype(mx.float32)
        x = x.reshape(bsize, length, self.num_heads, -1)
        B, C, dt = mx.split(BCdt, [self.d_state, self.d_state * 2], axis=-1)
        B = B[:, :, None, :]
        C = C[:, :, None, :]

        A = -mx.exp(self.A_log.astype(mx.float32))  # (num_heads,)
        dt = _rms_norm(dt, None, self.config.rms_norm_eps) * self.dt_norm_weight[None, None, :]
        B = _rms_norm(B, None, self.config.rms_norm_eps) * self.B_norm_weight[None, None, None, :]
        C = _rms_norm(C, None, self.config.rms_norm_eps) * self.C_norm_weight[None, None, None, :]

        # (bsize, length, num_heads, 1)
        dt = self.dt_proj(dt)[..., None]

        # TODO it may not be required
        B = mx.broadcast_to(B, (B.shape[0], B.shape[1], self.num_heads, B.shape[3]))
        C = mx.broadcast_to(C, (C.shape[0], C.shape[1], self.num_heads, C.shape[3]))

        if bool_mask is not None:
            """
            state will be updates by following:
            ```
            dt = softplus(dt)
            dA = exp(dt * A)
            state_next = state * dA + dB * x
            ```
            To avoid updating state, we set dt to -inf and x to 0
            because `softplus(-inf) = 0` and `exp(0) = 1`
            """
            dt = mx.where(bool_mask[:, :, None, None], dt, float("-inf"))
            x = mx.where(bool_mask[:, :, None, None], x, 0.0)

        # ssm
        self.D = self.D.astype(mx.float32)
        self.dt_bias = self.dt_bias.astype(mx.float32)
        if is_update:
            assert ssm_state is not None
            out, ssm_state = ssd_update_state(
                ssm_state,
                x[:, 0],
                dt[:, 0].reshape(bsize, -1),
                A,
                B[:, 0],
                C[:, 0],
                D=self.D,
                z=z[:, 0],
                dt_bias=self.dt_bias,
                dt_softplus=True,
            )
        else:
            out, ssm_state = ssd_chunk_scan_combined(
                x,
                dt.reshape(bsize, length, -1),
                A,
                B,
                C,
                self.chunk_size,
                D=self.D,
                z=z,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                ssm_state=ssm_state,
            )

        y = self.out_proj(out.reshape(bsize, length, -1))

        if past_states is not None:
            assert ssm_state is not None
            assert conv_state is not None
            past_states.update_mamba(conv_state, ssm_state, self.layer_idx)

        return y, past_states


class MLP(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.gate_up_proj(x)
        h = _swiglu(h)
        return self.down_proj(h)  # type: ignore


class PlamoDecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, is_mamba: bool, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.is_mamba = is_mamba
        self.mixer: nn.Module
        if is_mamba:
            self.mixer = Mamba(config, layer_idx)
        else:
            self.mixer = Attention(config, layer_idx)
        self.mlp = MLP(config)
        """
        Notes: The model performance was degraded when setting all offsets to 1.
        """
        self.pre_mixer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, offset=1.0)
        self.post_mixer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, offset=1.0 / 5)
        self.pre_mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, offset=1.0)
        self.post_mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, offset=1.0 / (5**1.5))

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        past_state: Optional[PlamoCache] = None,
        output_attentions: Optional[bool] = False,
    ) -> tuple[Any, ...]:
        # from LlamaDecoder
        residual = hidden_states
        hidden_states = self.pre_mixer_norm(hidden_states)

        # Self Attention
        if self.is_mamba:
            hidden_states_sa, present_key_value = self.mixer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_states=past_state,
            )
            self_attn_weights = None
        else:
            hidden_states_sa, self_attn_weights, present_key_value = self.mixer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_states=past_state,
                output_attentions=output_attentions,
            )

        hidden_states_sa = self.post_mixer_norm(hidden_states_sa)
        hidden_states = residual + hidden_states_sa

        residual = hidden_states
        hidden_states = self.pre_mlp_norm(hidden_states)

        # Fully Connected
        hidden_states_mlp = self.mlp(hidden_states)

        # Residual
        hidden_states_mlp = self.post_mlp_norm(hidden_states_mlp)
        hidden_states = residual + hidden_states_mlp

        outputs: Any = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs  # type: ignore


class PlamoDecoder(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()

        self.layers = [
            PlamoDecoderLayer(config, is_mamba=is_mamba(config, i), layer_idx=i)
            for i in range(config.num_hidden_layers)
        ]
        self.gradient_checkpointing = False

    def __call__(self, x: DecoderInput) -> DecoderOutput:
        all_hidden_states: Optional[tuple[mx.array, ...]] = () if x.output_hidden_states else None
        all_self_attns: Optional[tuple[mx.array, ...]] = () if x.output_attentions else None
        hidden_states = x.hidden_states

        for decoder_layer in self.layers:
            if x.output_hidden_states:
                assert all_hidden_states is not None
                all_hidden_states += (hidden_states,)

            if self.training and x.gradient_checkpointing:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    x.attention_mask,
                    x.past_states,
                    x.output_attentions,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=x.attention_mask,
                    past_state=x.past_states,
                    output_attentions=x.output_attentions,
                )

            hidden_states = layer_outputs[0]

            if x.output_attentions:
                assert layer_outputs[1] is not None
                assert all_self_attns is not None
                all_self_attns += (layer_outputs[1],)
        return DecoderOutput(hidden_states, all_hidden_states, all_self_attns)


class ModelOutput(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def to_tuple(self) -> tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())


class BaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (:obj:`mx.array` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If :obj:`past_key_values` is used only the last hidden-state of the sequences of shape
            :obj:`(batch_size, 1, hidden_size)` is output.
        past_key_values (:obj:`list[mx.array]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            list of :obj:`mx.array` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            ``past_key_values`` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(mx.array)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`mx.array` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(mx.array)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`mx.array` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_hidden_state: mx.array = kwargs.pop("last_hidden_state")
        self.past_key_values: Optional[tuple[tuple[mx.array]]] = kwargs.pop("past_key_values", None)
        self.hidden_states: Optional[tuple[mx.array, ...]] = kwargs.pop("hidden_states", None)
        self.attentions: Optional[tuple[mx.array, ...]] = kwargs.pop("attentions", None)


class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`mx.array` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`mx.array` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(mx.array))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(mx.array)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(mx.array)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mx.array` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(mx.array)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mx.array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.loss: Optional[mx.array] = kwargs.pop("loss", None)
        self.logits: mx.array | None = kwargs.pop("logits", None)
        self.past_key_values: Optional[tuple[tuple[mx.array]]] = kwargs.pop("past_key_values", None)
        self.hidden_states: Optional[tuple[mx.array, ...]] = kwargs.pop("hidden_states", None)
        self.attentions: Optional[tuple[mx.array, ...]] = kwargs.pop("attentions", None)


class PlamoPreTrainedModel(nn.Module):  # type: ignore
    config_class = ModelArgs
    _no_split_modules: list[str]
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PlamoDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

    def _init_weights(self, module: nn.Module) -> None:
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight = mx.random.normal(loc=0.0, scale=std, shape=module.weight.shape)
            if module.bias is not None:
                module.bias = mx.zeros_like(module.bias)
        elif isinstance(module, nn.Embedding):
            module.weight = mx.random.normal(loc=0.0, scale=std, shape=module.weight.shape)
            if module.padding_idx is not None:
                module.weight[module.padding_idx] = mx.zeros_like(module.weight[module.padding_idx])


class PlamoModel(PlamoPreTrainedModel):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        assert config.eval_attention_n_bit is None
        assert config.eval_mlp_n_bit is None

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = PlamoDecoder(config)  # type: ignore
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(
        self,
        attention_mask: mx.array,
        input_shape: tuple[int, int],
        inputs_embeds: Optional[mx.array],
        past_key_values_length: int,
    ) -> Optional[mx.array]:
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask: Optional[mx.array] = None
        if input_shape[-1] > 1:
            assert inputs_embeds is not None
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                past_key_values_length=past_key_values_length,
            )
            input_shape = (input_shape[0], combined_attention_mask.shape[2])

        if attention_mask is not None:
            if attention_mask.ndim == 4:
                # Custom 4D attention mask
                expanded_attn_mask = attention_mask
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                assert inputs_embeds is not None
                expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_values: Optional[PlamoCache] = None,
        inputs_embeds: Optional[mx.array] = None,
        image_features: Optional[mx.array] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        assert input_ids is not None
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values.get_seq_length()
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if image_features is not None:
            assert self.config.image_token_id is not None
            image_embeds = self.image_proj(image_features)
            assert image_embeds.shape == inputs_embeds.shape, (
                image_embeds.shape,
                inputs_embeds.shape,
            )
            mask = input_ids == self.config.image_token_id
            inputs_embeds[mask] = image_embeds[mask]

        # embed positions
        require_attn_mask = False
        if not self.training or past_key_values is not None:
            require_attn_mask = True
        if seq_length_with_past >= self.config.attention_window_size:
            require_attn_mask = True
        if require_attn_mask and attention_mask is None:
            attention_mask = mx.ones(
                (batch_size, seq_length_with_past),
                dtype=mx.bool_,  # type: ignore
            )
        if attention_mask is not None:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = PlamoCache(self.config)

        # decoder layers
        out = self.layers(
            DecoderInput(
                hidden_states,
                attention_mask,
                past_key_values,
                output_hidden_states,
                output_attentions,
                self.gradient_checkpointing,
            )
        )
        assert isinstance(out, DecoderOutput)
        hidden_states = out.hidden_states
        all_hidden_states = out.all_hidden_states
        all_self_attns = out.all_self_attns

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            assert all_hidden_states is not None
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Model(PlamoPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    # Without this, the model cannot be loaded into a meta device.
    # Relevant code:
    # https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/modeling_utils.py#L4376-L4381
    # https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/modeling_utils.py#L356
    # https://github.com/pytorch/pytorch/blob/v2.4.1/torch/nn/modules/module.py#L2068
    _supports_param_buffer_assignment = False

    def __init__(self, config: ModelArgs) -> None:
        super().__init__(config)
        self.config = config
        self.model = PlamoModel(config)

        self.vocab_size = config.vocab_size
        vocab_size = ((self.vocab_size + 15) // 16) * 16

        if not config.tie_word_embeddings:
            self.lm_head: nn.Module = nn.Linear(config.hidden_size, vocab_size, bias=False)

        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def set_decoder(self, decoder: PlamoModel) -> None:
        self.model = decoder

    def get_decoder(self) -> PlamoModel:
        return self.model

    def sanitize(self, weights: dict[Any, Any]) -> dict[Any, Any]:
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
        return weights

    def make_cache(self) -> PlamoCache:
        return PlamoCache(self.config)

    def __call__(self, inputs: mx.array, cache: PlamoCache | None = None) -> mx.array:
        model_inputs = self.prepare_inputs_for_generation(
            input_ids=inputs,
            past_key_values=cache,
            use_cache=self.config.use_cache,
        )
        output = self.forward(**model_inputs)
        if not isinstance(output, CausalLMOutputWithPast):
            raise ValueError(
                f"Unexpected output type for causal language model: {type(output)} != CausalLMOutputWithPast"
            )
        if output.past_key_values is not None:
            # output.past_key_values is actually a PlamoCache object
            self._cache = output.past_key_values  # type: ignore
        if output.logits is not None:
            return output.logits
        else:
            raise ValueError("The model did not return any logits.")

    def forward(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_values: Optional[PlamoCache] = None,
        inputs_embeds: Optional[mx.array] = None,
        image_features: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[Any, ...], CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`mx.array` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        Example:
        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM
        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""
        assert input_ids is not None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            image_features=image_features,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        elif isinstance(outputs, BaseModelOutputWithPast):
            hidden_states = outputs.last_hidden_state

        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(hidden_states)
        else:
            logits = self.lm_head(hidden_states)

        logits = logits[..., : self.vocab_size]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss_fct = nn.losses.cross_entropy
            shift_logits = shift_logits.reshape((-1, self.config.vocab_size))
            shift_labels = shift_labels.reshape((-1,))
            # Enable model parallelism
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if not isinstance(outputs, BaseModelOutputWithPast):
            raise ValueError(
                f"Unexpected output type for causal language model: {type(outputs)} != BaseModelOutputWithPast"
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: mx.array,
        past_key_values: Optional[PlamoCache] = None,
        attention_mask: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        image_features: Optional[mx.array] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if past_key_values:
            input_ids = input_ids[:, -1:]
            if image_features is not None:
                image_features = image_features[:, -1:, :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.astype(mx.int64).cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs: dict[str, Any] = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "image_features": image_features,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values: PlamoCache, beam_idx: mx.array) -> PlamoCache:
        past_key_values.reorder_cache(beam_idx)
        return past_key_values

    @property
    def layers(self):
        return self.model.layers
