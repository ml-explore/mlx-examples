# Copyright © 2023-2024 Apple Inc.

import math
from dataclasses import dataclass
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "phi"
    max_position_embeddings: int = 2048
    vocab_size: int = 51200
    hidden_size: int = 2560
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: int = 32
    partial_rotary_factor: float = 0.4
    intermediate_size: int = 10240
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


class PhiAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.repeats = self.num_heads // self.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.dense = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=True
        )

        self.rope = nn.RoPE(
            int(self.partial_rotary_factor * self.head_dim),
            traditional=False,
            base=self.rope_theta,
        )

    def __call__(self, x, mask=None, cache=None):
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Extract some shapes
        B, L, D = queries.shape
        n_heads, n_kv_heads = self.num_heads, self.num_key_value_heads

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(
            B,
            L,
            n_heads,
            -1,
        ).moveaxis(1, 2)
        keys = keys.reshape(B, L, n_kv_heads, -1).moveaxis(1, 2)
        values = values.reshape(B, L, n_kv_heads, -1).moveaxis(1, 2)

        # Add RoPE to the queries and keys and combine them with the cache
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scale = math.sqrt(1 / queries.shape[-1])
        output = mx.fast.scaled_dot_product_attention(
            queries.astype(mx.float32), keys, values, scale=scale, mask=mask
        ).astype(values.dtype)

        output = output.moveaxis(2, 1).reshape(B, L, -1)

        return self.dense(output)


class PhiMLP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = nn.GELU(approx="precise")

    def __call__(self, x) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class PhiDecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.self_attn = PhiAttention(config=config)
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.mlp = PhiMLP(config)

    def __call__(self, x, mask, cache):
        h = self.input_layernorm(x)
        attn_h = self.self_attn(h, mask, cache)
        ff_h = self.mlp(h)
        return attn_h + ff_h + x


class PhiModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [PhiDecoderLayer(config) for i in range(config.num_hidden_layers)]
        self.final_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def __call__(self, x, cache):
        x = self.embed_tokens(x)

        mask = create_attention_mask(x, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            x = layer(x, mask, c)
        return self.final_layernorm(x)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.model_type = config.model_type
        self.model = PhiModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        self.args = config

    def __call__(
        self,
        x: mx.array,
        cache: mx.array = None,
    ) -> Tuple[mx.array, mx.array]:
        y = self.model(x, cache)
        return self.lm_head(y)

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
