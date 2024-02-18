from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import BaseModelArgs
from .layers import RMSNorm


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    n_shared_head: int = (8,)
    rope_theta: float = 10000
    rope_traditional: bool = False


class Attention(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        head_dim = self.hidden_size // config.num_attention_heads

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
        self.rotary_emb = nn.RoPE(
            head_dim,
            traditional=config.rope_traditional,
            base=config.rope_theta,
            scale=1.0,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
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

        kv_seq_len = 0
        if cache is not None:
            kv_seq_len += cache[0].shape[-2]
        query_states = self.rotary_emb(query_states, offset=kv_seq_len)
        key_states = self.rotary_emb(key_states, offset=kv_seq_len)

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

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))  # type: ignore


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
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[Any, ...]:
        # from LlamaDecoder
        residual = hidden_states

        hidden_states = self.norm(hidden_states)

        # Self Attention
        hidden_states_sa, cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            cache=cache,
        )

        # Fully Connected
        hidden_states_mlp = self.mlp(hidden_states)

        hidden_states = residual + hidden_states_sa + hidden_states_mlp
        return hidden_states, cache


class PlamoDecoder(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.layers = [
            PlamoDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]


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

        for e, layer in enumerate(self.layers.layers):
            h, c = layer(h, mask, cache[e])
            if cache is not None:
                cache[e] = c
            else:
                cache.append(c)

        return self.norm(h), cache


class Model(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.model_type = args.model_type
        self.model = PlamoModel(args)
        self.lm_head: nn.Module = nn.Linear(
            args.hidden_size, args.vocab_size, bias=False
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, mx.array]:
        out, cache = self.model(inputs, cache)
        return self.lm_head(out), cache
