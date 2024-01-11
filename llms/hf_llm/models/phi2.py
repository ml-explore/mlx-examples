import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    n_positions: int = 2048
    vocab_size: int = 51200
    n_embd: int = 2560
    n_head: int = 32
    n_layer: int = 32
    rotary_dim: int = 32


class LayerNorm(nn.LayerNorm):
    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x.astype(mx.float32)).astype(x.dtype)


class RoPEAttention(nn.Module):
    def __init__(self, dims: int, n_head: int, rotary_dim: int):
        super().__init__()

        self.n_head = n_head

        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.dense = nn.Linear(dims, dims)

        self.rope = nn.RoPE(rotary_dim, traditional=False)

    def __call__(self, x, mask=None, cache=None):
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Extract some shapes
        n_head = self.n_head
        B, L, D = queries.shape

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, n_head, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, n_head, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, n_head, -1).transpose(0, 2, 1, 3)

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

        return self.dense(values_hat), (keys, values)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU(approx="precise")

    def __call__(self, x) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class ParallelBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        dims = config.n_embd
        mlp_dims = dims * 4
        self.self_attn = RoPEAttention(dims, config.n_head, config.rotary_dim)
        self.input_layernorm = LayerNorm(dims)
        self.mlp = MLP(dims, mlp_dims)

    def __call__(self, x, mask, cache):
        h = self.input_layernorm(x)
        attn_h, cache = self.self_attn(h, mask, cache)
        ff_h = self.mlp(h)
        return attn_h + ff_h + x, cache


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = [ParallelBlock(config) for i in range(config.n_layer)]
        self.final_layernorm = LayerNorm(config.n_embd)

    def __call__(self, x, mask, cache):
        x = self.embed_tokens(x)
        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            x, cache[e] = layer(x, mask, cache[e])
        return self.final_layernorm(x), cache


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.model = Transformer(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array = None,
        cache: mx.array = None,
    ) -> tuple[mx.array, mx.array]:
        mask = None
        if x.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
            mask = mask.astype(x.dtype)

        y, cache = self.model(x, mask, cache)
        return self.lm_head(y), cache
