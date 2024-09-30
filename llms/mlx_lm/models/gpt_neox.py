# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import BaseModelArgs, create_attention_mask

# Based on the transformers implementation at:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    max_position_embeddings: int
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    layer_norm_eps: float
    vocab_size: int
    rotary_emb_base: int
    rotary_pct: float
    num_key_value_heads: int = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert (
            args.hidden_size % args.num_attention_heads == 0
        ), "hidden_size must be divisible by num_attention_heads"

        self.hidden_size = args.hidden_size
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.rope = nn.RoPE(
            dims=int(self.head_dim * args.rotary_pct),
            traditional=False,
            base=args.rotary_emb_base,
        )

        self.scale = self.head_dim**-0.5

        self.query_key_value = nn.Linear(
            self.hidden_size, 3 * self.hidden_size, bias=True
        )
        self.dense = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qkv = self.query_key_value(x)

        new_qkv_shape = qkv.shape[:-1] + (self.num_attention_heads, 3 * self.head_dim)
        qkv = qkv.reshape(*new_qkv_shape)

        queries, keys, values = [x.transpose(0, 2, 1, 3) for x in qkv.split(3, -1)]

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.dense(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.dense_h_to_4h = nn.Linear(self.hidden_size, 4 * self.hidden_size)
        self.dense_4h_to_h = nn.Linear(4 * self.hidden_size, self.hidden_size)

    def __call__(self, x) -> mx.array:
        # gelu_approx corresponds to FastGELUActivation in transformers.
        return self.dense_4h_to_h(nn.gelu_approx(self.dense_h_to_4h(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.layer_norm_eps = args.layer_norm_eps
        self.attention = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.LayerNorm(
            self.hidden_size,
            eps=self.layer_norm_eps,
        )
        self.post_attention_layernorm = nn.LayerNorm(
            self.hidden_size, eps=self.layer_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        residual = x
        # NeoX runs attention and feedforward network in parallel.
        attn = self.attention(self.input_layernorm(x), mask, cache)
        ffn = self.mlp(self.post_attention_layernorm(x))
        out = attn + ffn + residual
        return out


class GPTNeoXModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.layer_norm_eps = args.layer_norm_eps
        assert self.vocab_size > 0
        self.embed_in = nn.Embedding(self.vocab_size, self.hidden_size)
        self.embed_out = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.h = [TransformerBlock(args=args) for _ in range(self.num_hidden_layers)]
        self.final_layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        _, L = inputs.shape

        hidden_states = self.embed_in(inputs)

        mask = create_attention_mask(hidden_states, cache)

        if cache is None:
            cache = [None] * len(self.h)

        for layer, c in zip(self.h, cache):
            hidden_states = layer(hidden_states, mask, cache=c)

        out = self.final_layer_norm(hidden_states)
        out = self.embed_out(out)

        return out


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GPTNeoXModel(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        return out

    def sanitize(self, weights):
        new_weights = {}

        for w_key, w_value in weights.items():
            # Created through register_buffer in Pytorch, not needed here.
            ignore_suffixes = [
                ".attention.bias",
                ".attention.masked_bias",
                ".attention.rotary_emb.inv_freq",
            ]

            skip_weight = False
            for ignored_suffix in ignore_suffixes:
                if w_key.endswith(ignored_suffix):
                    skip_weight = True
                    break

            if skip_weight:
                continue

            if not w_key.startswith("model."):
                w_key = f"model.{w_key}"

            w_key = w_key.replace(".gpt_neox.layers.", ".h.")
            w_key = w_key.replace(".gpt_neox.", ".")

            new_weights[w_key] = w_value

        return new_weights

    @property
    def layers(self):
        return self.model.h

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
