import math
from dataclasses import dataclass
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    intermediate_size: int
    rope_theta: float
    use_qkv_bias: bool
    partial_rotary_factor: float
    layer_norm_eps: float
    use_parallel_residual: bool = False
    qk_layernorm: bool = False


class LayerNormPerHead(nn.Module):

    def __init__(self, head_dim, num_heads, eps):
        super().__init__()
        self.norms = [
            nn.LayerNorm(head_dim, eps=eps, bias=False) for _ in range(num_heads)
        ]
        self.eps = eps

    def __call__(self, x):
        w = mx.stack([n.weight for n in self.norms])
        return w * mx.fast.layer_norm(x, None, None, self.eps)


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor

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
            int(self.partial_rotary_factor * self.head_dim),
            traditional=False,
            base=self.rope_theta,
        )

        self.qk_layernorm = config.qk_layernorm
        if self.qk_layernorm:
            self.q_layernorm = LayerNormPerHead(
                self.head_dim, self.num_heads, eps=config.layer_norm_eps
            )
            self.k_layernorm = LayerNormPerHead(
                self.head_dim, self.num_key_value_heads, eps=config.layer_norm_eps
            )

    def __call__(self, x, mask=None, cache=None):
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Extract some shapes
        B, L, D = queries.shape

        queries = queries.reshape(B, L, self.num_heads, -1)
        keys = keys.reshape(B, L, self.num_key_value_heads, -1)
        if self.qk_layernorm:
            queries = self.q_layernorm(queries)
            keys = self.k_layernorm(keys)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        # Add RoPE to the queries and keys and combine them with the cache
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        queries = queries.astype(mx.float32)
        keys = keys.astype(mx.float32)

        # Finally perform the attention computation
        scale = math.sqrt(1 / queries.shape[-1])
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask
        ).astype(values.dtype)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


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
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.use_parallel_residual = config.use_parallel_residual
        if not self.use_parallel_residual:
            self.post_attention_layernorm = nn.LayerNorm(
                config.hidden_size,
                eps=config.layer_norm_eps,
            )

    def __call__(self, x, mask, cache):
        h = self.input_layernorm(x)
        r = self.self_attn(h, mask, cache)

        if self.use_parallel_residual:
            out = x + r + self.mlp(h)
        else:
            h = x + r
            r = self.mlp(self.post_attention_layernorm(h))
            out = h + r
        return out


class StableLM(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config) for i in range(config.num_hidden_layers)]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x, mask, cache):
        x = self.embed_tokens(x)
        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            x = layer(x, mask, cache=c)

        return self.norm(x)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.model_type = config.model_type
        self.model = StableLM(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.args = config

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

        y = self.model(x, mask, cache)
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
