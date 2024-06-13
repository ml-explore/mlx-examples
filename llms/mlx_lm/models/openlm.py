from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import BaseModelArgs, create_additive_causal_mask


@dataclass
class ParamsArgs(BaseModelArgs):
    dim: int
    ffn_type: str
    n_heads: int
    n_layers: int
    norm_eps: float
    positional_embedding_type: str
    post_embed_norm: bool
    qk_norm: bool
    vocab_size: int
    weight_tying: bool


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    params_args_dict: ParamsArgs


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.dim // self.n_heads
        self.qk_norm = args.qk_norm
        self.scale = self.head_dim**-0.5

        self.in_proj = nn.Linear(self.dim, 3 * self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        if self.qk_norm:
            self.q_norm = nn.LayerNorm(args.dim, eps=args.norm_eps, bias=False)
            self.k_norm = nn.LayerNorm(args.dim, eps=args.norm_eps, bias=False)
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=False,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.in_proj(x).split(3, axis=-1)

        if self.qk_norm:
            queries = self.q_norm(queries)
            keys = self.q_norm(keys)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)

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
        return self.out_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # https://github.com/mlfoundations/open_lm/blob/c65b43042ff31c0fe26f930decf1ccab1b03ab4b/open_lm/model.py#L254C2-L254C3
        hidden_dim = 256 * ((int(2 * 4 * args.dim / 3) + 256 - 1) // 256)
        self.w12 = nn.Linear(args.dim, 2 * hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, args.dim, bias=False)

    def __call__(self, x) -> mx.array:
        gate, x = self.w12(x).split(2, axis=-1)
        return self.w3(nn.silu(gate) * x)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = MLP(args)
        self.ffn_norm = nn.LayerNorm(args.dim, eps=args.norm_eps, bias=False)
        self.attention_norm = nn.LayerNorm(args.dim, eps=args.norm_eps, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out


class OpenLM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        self.norm = nn.LayerNorm(args.dim, eps=args.norm_eps, bias=False)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        _, L = inputs.shape

        h = self.tok_embeddings(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = create_additive_causal_mask(
                h.shape[1], cache[0].offset if cache is not None else 0
            )
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.output(self.norm(h))


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        args.params_args_dict = ParamsArgs.from_dict(args.params_args_dict)
        self.args = args.params_args_dict
        self.model_type = args.model_type
        self.model = OpenLM(self.args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        return out

    def sanitize(self, weights):
        # Remove unused precomputed rotary freqs
        return {k: v for k, v in weights.items() if "inv_freq" not in k}

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.dim // self.args.n_heads

    @property
    def n_kv_heads(self):
        return self.args.n_heads
