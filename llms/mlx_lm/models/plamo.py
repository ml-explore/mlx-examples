from typing import Any, List, NamedTuple, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import PretrainedConfig


class DecoderInput(NamedTuple):
    hidden_states: mx.array
    position_ids: mx.array
    attention_mask: Optional[mx.array] = None
    past_key_values: Optional[List[mx.array]] = None
    output_hidden_states: Optional[bool] = False
    output_attentions: Optional[bool] = False
    use_cache: Optional[bool] = False
    gradient_checkpointing: bool = False


class DecoderOutput(NamedTuple):
    hidden_states: mx.array
    all_hidden_states: Optional[Tuple[mx.array, ...]]
    all_self_attns: Optional[Tuple[mx.array, ...]]
    next_decoder_cache: Optional[Tuple[mx.array, ...]]


class ModelArgs(PretrainedConfig):  # type: ignore
    model_type: str = "plamo"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 13312,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tokenizer_class: str = "PlamoTokenizer",
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        n_shared_head: int = 8,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.n_shared_head = n_shared_head

        super().__init__(
            tokenizer_class=tokenizer_class,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class RotaryEmbedding:
    def __init__(
        self, dim: int, max_position_embeddings: int = 2048, base: int = 10000
    ) -> None:
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / mx.power(
            self.base, mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim
        )
        self.cos_cached = mx.zeros((1, 1, max_position_embeddings, dim))
        self.sin_cached = mx.zeros((1, 1, max_position_embeddings, dim))
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        self.max_seq_len_cached = seq_len
        t = mx.arange(self.max_seq_len_cached)  # type: ignore

        freqs = mx.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = mx.concatenate((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def __call__(self, x: mx.array, seq_len: int) -> Tuple[mx.array, mx.array]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        return (
            self.cos_cached[:, :, :seq_len, ...].astype(x.dtype),  # type: ignore
            self.sin_cached[:, :, :seq_len, ...].astype(x.dtype),  # type: ignore
        )


def _rotate_half(x: mx.array) -> mx.array:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate((-x2, x1), axis=-1)


def _rotary_pos_emb(
    x: mx.array, cos: mx.array, sin: mx.array, position_ids: mx.array
) -> mx.array:
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = mx.squeeze(cos, (0, 1))  # [seq_len, dim]
    sin = mx.squeeze(sin, (0, 1))  # [seq_len, dim]
    cos = cos[position_ids][:, None]  # [bs, 1, seq_len, dim]
    sin = sin[position_ids][:, None]  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (_rotate_half(x) * sin)
    return x_embed


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.variance_epsilon = eps

    def _norm(self, x):
        return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + self.variance_epsilon)

    def __call__(self, x):
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output


class Attention(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        head_dim = self.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.q_num_heads = config.num_attention_heads
        self.qk_dim = self.v_dim = head_dim
        self.k_num_heads = self.v_num_heads = int(
            np.ceil(self.q_num_heads / config.n_shared_head)
        )

        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.q_num_heads * self.qk_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.k_num_heads * self.qk_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.v_num_heads * self.v_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.q_num_heads * self.v_dim, self.hidden_size, bias=False
        )
        self.rotary_emb = RotaryEmbedding(
            self.qk_dim, max_position_embeddings=self.max_position_embeddings
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Prepare the queries, keys and values for the attention computation
        query_states = query_states.reshape(
            bsz, q_len, self.q_num_heads, self.qk_dim
        ).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(
            bsz, q_len, self.k_num_heads, self.qk_dim
        ).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(
            bsz, q_len, self.v_num_heads, self.v_dim
        ).transpose(0, 2, 1, 3)

        def _expand_kv(a: mx.array) -> mx.array:
            a = mx.concatenate(
                [mx.expand_dims(a, 1)] * self.config.n_shared_head, axis=1
            )
            return a.reshape([bsz, self.q_num_heads, q_len, -1])

        # expand shared kv
        assert self.k_num_heads == self.v_num_heads
        key_states = _expand_kv(key_states)
        value_states = _expand_kv(value_states)

        kv_seq_len = key_states.shape[-2]
        if cache is not None:
            kv_seq_len += cache[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        assert position_ids is not None
        query_states = _rotary_pos_emb(query_states, cos, sin, position_ids)
        key_states = _rotary_pos_emb(key_states, cos, sin, position_ids)

        if cache is not None:
            # reuse k, v, self_attention
            key_states = mx.concatenate([cache[0], key_states], axis=2)
            value_states = mx.concatenate([cache[1], value_states], axis=2)

        scores = (query_states * self.scale) @ key_states.transpose(0, 1, 3, 2)
        if attention_mask is not None:
            scores += attention_mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ value_states).transpose(0, 2, 1, 3).reshape(bsz, q_len, -1)

        return self.o_proj(output), (key_states, value_states)


class MLP(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.silu

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))  # type: ignore


class PlamoDecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[Any, ...]:
        # from LlamaDecoder
        residual = hidden_states

        hidden_states = self.norm(hidden_states)

        # Self Attention
        hidden_states_sa, cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
        )

        # Fully Connected
        hidden_states_mlp = self.mlp(hidden_states)

        # Residual ("Parallel Layers" is used here, which is different from the normal residual connection)
        # See "GPT-NeoX-20B: An Open-Source Autoregressive Language Model" for Parallel Layers
        hidden_states = residual + hidden_states_sa + hidden_states_mlp

        return hidden_states, cache  # type: ignore


class PlamoDecoder(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.layers = [
            PlamoDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]


class PlamoModel(nn.Module):
    config_class = ModelArgs
    _no_split_modules: List[str]
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PlamoDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = PlamoDecoder(config)  # type: ignore
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Union[Tuple[mx.array, mx.array], None]]] = None,
    ) -> Tuple[mx.array, Optional[List[Union[Tuple[mx.array, mx.array], None]]]]:
        h = self.embed_tokens(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(self.embed_tokens.weight.dtype)

        if cache is None:
            past_key_values_length = 0
            cache = [None for _ in range(len(self.layers.layers))]
        else:
            if cache[0] is not None:
                past_key_values_length = cache[0][0].shape[2]
        position_ids = _create_position_ids(h.shape[1], past_key_values_length)

        for e, layer in enumerate(self.layers.layers):
            h, c = layer(h, mask, position_ids, cache[e])
            if cache is not None:
                cache[e] = c
            else:
                cache.append(c)

        return self.norm(h), cache


def _create_position_ids(seq_length: int, past_key_values_length: int = 0) -> mx.array:
    # create position_ids on the fly for batch generation
    position_ids = mx.arange(
        past_key_values_length, seq_length + past_key_values_length, dtype=mx.int64
    )
    position_ids = position_ids[None, ...].reshape(-1, seq_length)

    return position_ids


class Model(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.model = PlamoModel(config)
        self.lm_head: nn.Module = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, mx.array]:
        out, cache = self.model(inputs, cache)
        return self.lm_head(out), cache
