from dataclasses import dataclass
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    kv_channels: int = 128
    max_position_embeddings: int = 8192
    layer_norm_epsilon: float = 1e-6
    intermediate_size: int = 11008
    no_bias: bool = True
    vocab_size: int = 151936
    num_key_value_heads = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_size = args.hidden_size
        self.num_attention_heads = args.num_attention_heads

        hidden_size_per_attention_head = hidden_size // self.num_attention_heads

        self.rotary_emb = nn.RoPE(hidden_size_per_attention_head, traditional=False)

        proj_size = args.kv_channels * self.num_attention_heads

        self.c_attn = nn.Linear(hidden_size, proj_size * 3, bias=True)
        self.c_proj = nn.Linear(hidden_size, proj_size, bias=not args.no_bias)

        self.scale = hidden_size_per_attention_head**-0.5

    def __call__(self, x, mask=None, cache=None):
        qkv = self.c_attn(x)

        q, k, v = mx.split(qkv, 3, axis=-1)

        B, L, _ = q.shape

        queries = q.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
        keys = k.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
        values = v.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rotary_emb(queries, offset=cache.offset)
            keys = self.rotary_emb(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rotary_emb(queries)
            keys = self.rotary_emb(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.c_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(
            args.hidden_size, args.intermediate_size // 2, bias=not args.no_bias
        )
        self.w2 = nn.Linear(
            args.hidden_size, args.intermediate_size // 2, bias=not args.no_bias
        )
        self.c_proj = nn.Linear(
            args.intermediate_size // 2, args.hidden_size, bias=not args.no_bias
        )

    def __call__(self, x):
        a1 = self.w1(x)
        a2 = self.w2(x)
        return self.c_proj(a1 * nn.silu(a2))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.ln_1 = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.attn = Attention(args)
        self.ln_2 = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.mlp = MLP(args)

    def __call__(self, x, mask=None, cache=None):
        residual = x
        x = self.ln_1(x)
        x = self.attn(x, mask=mask, cache=cache)
        residual = x + residual
        x = self.ln_2(residual)
        x = self.mlp(x)
        x = x + residual

        return x


class QwenModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.wte = nn.Embedding(args.vocab_size, args.hidden_size)
        self.h = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.ln_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

    def __call__(self, inputs, mask=None, cache=None):
        x = self.wte(inputs)

        mask = None
        T = x.shape[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(x.dtype)

        if cache is None:
            cache = [None] * len(self.h)

        for layer, c in zip(self.h, cache):
            x = layer(x, mask, c)

        return self.ln_f(x)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.model_type = config.model_type
        self.transformer = QwenModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=not config.no_bias
        )
        self.args = config

    def __call__(
        self,
        x: mx.array,
        mask: mx.array = None,
        cache: mx.array = None,
    ) -> Tuple[mx.array, mx.array]:
        y = self.transformer(x, mask, cache)
        return self.lm_head(y)

    @property
    def layers(self):
        return self.transformer.h

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_attention_heads
