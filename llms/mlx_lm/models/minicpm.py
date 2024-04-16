from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import BaseModelArgs

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    dim_model_base: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    rope_theta: float
    max_position_embeddings: int
    scale_depth: float

class MiniCPMRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class MiniCPMMLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.intermediate_size = args.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.act_fn = nn.silu

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MiniCPMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.rope_theta = args.rope_theta

        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rope = nn.RoPE(
            self.head_dim,
            traditional=True,
            base=self.rope_theta,
        )


    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        ) -> mx.array:
            B, L, _ = x.shape

            queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

            queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            keys = keys.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
            values = values.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

            if cache is not None:
                key_cache, value_cache = cache

                queries = self.rope(queries, offset=key_cache.shape[2])
                keys = self.rope(keys, offset=key_cache.shape[2])

                keys = mx.concatenate([key_cache, keys], axis=2)
                values = mx.concatenate([value_cache, values], axis=2)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)

            attn_output = mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=self.scale, mask=mask
            )

            # attn_output = attn_output.reshape(B, L, self.hidden_size) # Originale impementation
            attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)

            return self.o_proj(attn_output), (keys, values)


class MiniCPMDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.num_hidden_layers = args.num_hidden_layers

        self.self_attn = MiniCPMAttention(args)
        self.mlp = MiniCPMMLP(args)
        self.input_layernorm = MiniCPMRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = MiniCPMRMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.scale_depth = args.scale_depth

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r * (self.scale_depth / np.sqrt(self.num_hidden_layers))
        h = self.mlp(self.post_attention_layernorm(h))
        out = x + r * (self.scale_depth / np.sqrt(self.num_hidden_layers))
        return out, cache


class MiniCPMModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
                MiniCPMDecoderLayer(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = MiniCPMRMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        return self.norm(h), cache

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MiniCPMModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out, cache = self.model(inputs, cache)
        # out = out @ self.model.embed_tokens.weight.T
        out = self.lm_head(out / (self.args.hidden_size / self.args.dim_model_base))
        return out, cache

    def sanitize(self, weights):
        if "lm_head.weight" not in weights:
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
        # Remove unused precomputed rotary freqs
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

    @property
    def layers(self):
        return self.model.layers
