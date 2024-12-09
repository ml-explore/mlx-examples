# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float
    embed_dropout: float
    attention_dropout: float
    layer_norm_epsilon: float
    activation_function: str
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True
    attn_implementation: str = "eager"
    # For simplicity, we assume no bias in Q, K, V, and MLP similar to the original code
    attention_bias: bool = False
    mlp_bias: bool = False
        
    @classmethod
    def from_dict(cls, params):
        if 'num_layers' in params:
            params['num_hidden_layers'] = params['num_layers']
        if 'layer_norm_epsilon' in params:
            params['rms_norm_eps'] = params['layer_norm_epsilon']
        return super().from_dict(params)

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            rope_type = self.rope_scaling.get("type") or self.rope_scaling.get("rope_type")
            if rope_type is None:
                raise ValueError("rope_scaling must contain either 'type' or 'rope_type'")
            if rope_type not in ["linear", "dynamic", "llama3", "default"]:
                raise ValueError(
                    "rope_scaling 'type' currently only supports 'linear', 'dynamic', 'llama3', or 'default'"
                )


class ExaoneRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dims: int,
        max_position_embeddings: int = 2048,
        traditional: bool = False,
        base: float = 10000,
        scale: float = 1.0,
        rope_type: str = "default",
        rope_scaling: Optional[Dict[str, Union[float, str]]] = None,
    ):
        super().__init__()
        self.dims = dims
        self.max_position_embeddings = max_position_embeddings
        self.traditional = traditional
        self.scale = scale
        self.rope_type = rope_type
        self.rope_scaling = rope_scaling
        self.base = base

    def __call__(self, x, offset: int = 0):
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=self.base,
            scale=self.scale,
            offset=offset,
            freqs=None,
        )


def initialize_rope(args: ModelArgs):
    head_dim = args.head_dim or (args.hidden_size // args.num_attention_heads)
    rope_scaling = args.rope_scaling
    rope_type = "default"
    rope_scale = 1.0

    if rope_scaling is not None:
        rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type", "default")
        if rope_type == "linear":
            rope_scale = 1 / rope_scaling["factor"]
        elif rope_type in ["llama3", "dynamic"]:
            rope_scale = 1.0

    return ExaoneRotaryEmbedding(
        dims=head_dim,
        max_position_embeddings=args.max_position_embeddings or 2048,
        traditional=args.rope_traditional,
        base=args.rope_theta,
        scale=rope_scale,
        rope_type=rope_type,
        rope_scaling=rope_scaling,
    )


class AttentionModule(nn.Module):
    # This module corresponds to "attention" inside "attn"
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim or (dim // n_heads)
        self.scale = head_dim ** -0.5

        # Match naming exactly: q_proj, k_proj, v_proj, out_proj
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.out_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

        self.rope = initialize_rope(args)
        self.attention_dropout = args.attention_dropout

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)


        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.out_proj(out)


class Attention(nn.Module):
    # This corresponds to "attn" module that contains "attention"
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = AttentionModule(args)


class MLP(nn.Module):
    # This corresponds to "mlp" module that contains c_fc_0, c_fc_1, c_proj
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        self.c_fc_0 = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)
        self.c_fc_1 = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)
        self.c_proj = nn.Linear(hidden_dim, dim, bias=args.mlp_bias)

        

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(nn.silu(self.c_fc_0(x)) * self.c_fc_1(x))


class TransformerBlock(nn.Module):
    # A single layer: transformer.h.<layer>
    # contains: ln_1, attn, ln_2, mlp
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ln_1 = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attn = Attention(args)
        self.ln_2 = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.mlp = MLP(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.attn.attention(self.ln_1(x), mask, cache)
        out = h + self.mlp(self.ln_2(h))
        return out


class ExaoneModel(nn.Module):
    # top-level: transformer
    # contains: wte, h, ln_f
    def __init__(self, args: ModelArgs):
        super().__init__()
        # all these must be attributes of self.transformer to have "transformer." prefix
        self.wte = nn.Embedding(args.vocab_size, args.hidden_size)
        self.h = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.ln_f = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.embed_dropout = args.embed_dropout

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.wte(inputs)
        #h = nn.dropout(h, p=self.embed_dropout)
        mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.h)

        for (layer, c) in zip(self.h, cache):
            h = layer(h, mask, cache=c)

        return self.ln_f(h)


class Model(nn.Module):
    # The final model, containing `transformer` and optionally `lm_head`
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.transformer = ExaoneModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.transformer(inputs, cache)
        if self.args.tie_word_embeddings:
            # tie_word_embeddings means lm_head shares weight with wte
            out = self.transformer.wte.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        return {k: v for k, v in weights.items()}

    @property
    def layers(self):
        return self.transformer.h
