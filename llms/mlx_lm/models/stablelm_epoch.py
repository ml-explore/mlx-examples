import math
from dataclasses import dataclass
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs
from .layers import LayerNorm


@dataclass
class ModelArgs(BaseModelArgs):
    max_position_embeddings: int
    model_type: str
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rope_pct: float
    intermediate_size: int
    norm_eps: float
    rope_theta: float
    use_qkv_bias: bool


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.repeats = self.num_heads // self.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.rope_pct = config.rope_pct

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.use_qkv_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.use_qkv_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.use_qkv_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.rope = nn.RoPE(
            int(self.rope_pct * self.head_dim),
            traditional=False,
            base=self.rope_theta,
        )

    def __call__(self, x, mask=None, cache=None):
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Extract some shapes
        B, L, D = queries.shape

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(
            B, L, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        if self.repeats > 1:
            keys = mx.repeat(keys, self.repeats, axis=1)
            values = mx.repeat(values, self.repeats, axis=1)

        # Add RoPE to the queries and keys and combine them with the cache
        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        queries = queries.astype(mx.float32)
        keys = keys.astype(mx.float32)

        # Finally perform the attention computation
        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask

        scores = mx.softmax(scores, axis=-1).astype(values.dtype)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(values_hat), (keys, values)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.self_attn = Attention(config=config)
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = LayerNorm(
            config.hidden_size, eps=config.norm_eps
        )

    def __call__(self, x, mask, cache):
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache


class StableLM(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config) for i in range(config.num_hidden_layers)]
        self.norm = LayerNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(self, x, mask, cache):
        x = self.embed_tokens(x)
        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            x, cache[e] = layer(x, mask, cache[e])
        return self.norm(x), cache


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.model_type = config.model_type
        self.model = StableLM(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array = None,
        cache: mx.array = None,
    ) -> Tuple[mx.array, mx.array]:
        mask = None
        if x.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
            mask = mask.astype(x.dtype)

        y, cache = self.model(x, mask, cache)
        return self.lm_head(y), cache

    @property
    def layers(self):
        return self.model.layers
