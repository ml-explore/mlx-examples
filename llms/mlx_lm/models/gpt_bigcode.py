# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import BaseModelArgs, create_attention_mask


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    n_embd: int
    n_layer: int
    n_inner: int
    n_head: int
    n_positions: int
    layer_norm_epsilon: float
    vocab_size: int
    num_key_value_heads: int = None
    multi_query: bool = True
    attention_bias: bool = True
    mlp_bias: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = 1 if self.multi_query else self.n_head


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.dim = dim = args.n_embd
        self.n_heads = n_heads = args.n_head
        self.n_kv_heads = n_kv_heads = 1 if args.multi_query else args.n_head

        self.head_dim = head_dim = dim // n_heads

        self.kv_dim = n_kv_heads * head_dim

        self.scale = head_dim**-0.5

        if hasattr(args, "attention_bias"):
            attention_bias = args.attention_bias
        else:
            attention_bias = False

        self.c_attn = nn.Linear(dim, dim + 2 * self.kv_dim, bias=attention_bias)
        self.c_proj = nn.Linear(dim, dim, bias=attention_bias)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qkv = self.c_attn(x)
        queries, keys, values = mx.split(
            qkv, [self.dim, self.dim + self.kv_dim], axis=-1
        )

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.c_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.n_embd
        hidden_dim = args.n_inner
        if hasattr(args, "mlp_bias"):
            mlp_bias = args.mlp_bias
        else:
            mlp_bias = False

        self.c_fc = nn.Linear(dim, hidden_dim, bias=mlp_bias)
        self.c_proj = nn.Linear(hidden_dim, dim, bias=mlp_bias)

    def __call__(self, x) -> mx.array:
        return self.c_proj(nn.gelu(self.c_fc(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.attn = Attention(args)
        self.mlp = MLP(args)
        self.ln_1 = nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r = self.attn(self.ln_1(x), mask, cache)
        h = x + r
        r = self.mlp(self.ln_2(h))
        out = h + r
        return out


class GPTBigCodeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0
        self.wte = nn.Embedding(args.vocab_size, args.n_embd)
        self.wpe = nn.Embedding(args.n_positions, args.n_embd)
        self.h = [TransformerBlock(args=args) for _ in range(args.n_layer)]
        self.ln_f = nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        B, L = inputs.shape

        hidden_states = self.wte(inputs)

        mask = None
        if hidden_states.shape[1] > 1:

            position_ids = mx.array(np.arange(L))
            hidden_states += self.wpe(position_ids)

            mask = create_attention_mask(hidden_states, cache)

        if cache is None:
            cache = [None] * len(self.h)

        for layer, c in zip(self.h, cache):
            hidden_states = layer(hidden_states, mask, cache=c)

        return self.ln_f(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.transformer = GPTBigCodeModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.transformer(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.transformer.wte.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.transformer.h

    @property
    def head_dim(self):
        return self.args.n_embd // self.args.n_head

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
