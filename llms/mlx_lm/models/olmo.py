from dataclasses import dataclass
from sys import exit
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs

try:
    import hf_olmo
except ImportError:
    print("To run olmo install ai2-olmo: pip install ai2-olmo")
    exit(1)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    d_model: int
    n_layers: int
    mlp_hidden_size: int
    n_heads: int
    vocab_size: int
    embedding_size: int
    model_type: str
    rope_theta: float = 10000
    rope_traditional: bool = False
    mlp_ratio: int = 4
    weight_tying: bool = False

    def __post_init__(self):
        self.mlp_hidden_size = (
            self.mlp_hidden_size
            if self.mlp_hidden_size is not None
            else self.mlp_ratio * self.d_model
        )


class LayerNorm(nn.LayerNorm):
    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x.astype(mx.float32)).astype(x.dtype)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        dim = args.d_model

        self.ff_proj = nn.Linear(dim, args.mlp_hidden_size, bias=False)
        self.ff_out = nn.Linear(args.mlp_hidden_size // 2, dim, bias=False)

        self.att_norm = LayerNorm(dim, affine=False)
        self.ff_norm = LayerNorm(dim, affine=False)

        head_dim = dim // self.n_heads
        self.scale = head_dim**-0.5

        self.att_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.rope = nn.RoPE(
            head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
        )

        self.args = args

    def attend(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = mx.split(self.att_proj(x), 3, axis=-1)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.attn_out(output), (keys, values)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, cache = self.attend(self.att_norm(x), mask, cache)
        h = x + r

        x1, x2 = mx.split(self.ff_proj(self.ff_norm(h)), 2, axis=-1)

        out = h + self.ff_out(nn.silu(x2) * x1)
        return out, cache


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_layers = args.n_layers
        self.weight_tying = args.weight_tying

        self.wte = nn.Embedding(args.embedding_size, args.d_model)
        self.blocks = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        if not self.weight_tying:
            self.ff_out = nn.Linear(args.d_model, args.embedding_size, bias=False)
        self.norm = LayerNorm(args.d_model, affine=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.wte(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.blocks)

        for e, block in enumerate(self.blocks):
            h, cache[e] = block(h, mask, cache[e])

        h = self.norm(h)

        if self.weight_tying:
            return h @ self.wte.weight.T, cache

        return self.ff_out(h), cache


class OlmoModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.transformer = Transformer(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        return self.transformer(inputs, cache)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.model = OlmoModel(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        return self.model(inputs, cache)

    @property
    def layers(self):
        return self.model.transformer.blocks
