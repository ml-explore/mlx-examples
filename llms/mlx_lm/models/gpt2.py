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
    n_ctx: int
    n_embd: int
    n_head: int
    n_layer: int
    n_positions: int
    layer_norm_epsilon: float
    vocab_size: int
    num_key_value_heads: int = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.n_head


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.n_embd % args.n_head == 0, "n_embd must be divisible by n_head"

        self.n_embd = args.n_embd
        self.n_head = args.n_head
        self.head_dim = self.n_embd // self.n_head

        self.scale = self.head_dim**-0.5

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=True)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qkv = self.c_attn(x)
        queries, keys, values = mx.split(qkv, 3, axis=-1)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)

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

        self.n_embd = args.n_embd
        self.c_fc = nn.Linear(self.n_embd, 4 * self.n_embd)
        self.c_proj = nn.Linear(4 * self.n_embd, self.n_embd)

    def __call__(self, x) -> mx.array:
        return self.c_proj(nn.gelu_approx(self.c_fc(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.layer_norm_epsilon = args.layer_norm_epsilon
        self.attn = Attention(args)
        self.mlp = MLP(args)
        self.ln_1 = nn.LayerNorm(
            self.n_embd,
            eps=self.layer_norm_epsilon,
        )
        self.ln_2 = nn.LayerNorm(self.n_embd, eps=self.layer_norm_epsilon)

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


class GPT2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_embd = args.n_embd
        self.n_positions = args.n_positions
        self.vocab_size = args.vocab_size
        self.n_layer = args.n_layer
        self.layer_norm_epsilon = args.layer_norm_epsilon
        assert self.vocab_size > 0
        self.wte = nn.Embedding(self.vocab_size, self.n_embd)
        self.wpe = nn.Embedding(self.n_positions, self.n_embd)
        self.h = [TransformerBlock(args=args) for _ in range(self.n_layer)]
        self.ln_f = nn.LayerNorm(self.n_embd, eps=self.layer_norm_epsilon)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        _, L = inputs.shape

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
        self.model = GPT2Model(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        out = self.model.wte.as_linear(out)
        return out

    def sanitize(self, weights):
        new_weights = {}
        for i in range(self.args.n_layer):
            if f"h.{i}.attn.bias" in weights:
                del weights[f"h.{i}.attn.bias"]
            if f"h.{i}.attn.c_attn.weight" in weights:
                weights[f"h.{i}.attn.c_attn.weight"] = weights[
                    f"h.{i}.attn.c_attn.weight"
                ].transpose(1, 0)
            if f"h.{i}.attn.c_proj.weight" in weights:
                weights[f"h.{i}.attn.c_proj.weight"] = weights[
                    f"h.{i}.attn.c_proj.weight"
                ].transpose(1, 0)
            if f"h.{i}.mlp.c_fc.weight" in weights:
                weights[f"h.{i}.mlp.c_fc.weight"] = weights[
                    f"h.{i}.mlp.c_fc.weight"
                ].transpose(1, 0)
            if f"h.{i}.mlp.c_proj.weight" in weights:
                weights[f"h.{i}.mlp.c_proj.weight"] = weights[
                    f"h.{i}.mlp.c_proj.weight"
                ].transpose(1, 0)
        for weight in weights:
            if not weight.startswith("model."):
                new_weights[f"model.{weight}"] = weights[weight]
            else:
                new_weights[weight] = weights[weight]
        return new_weights

    @property
    def layers(self):
        return self.model.h

    @property
    def head_dim(self):
        return self.args.n_embd // self.args.n_head

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
