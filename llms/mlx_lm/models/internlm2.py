# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    bias: bool = True
    max_position_embeddings: int = 32768
    num_key_value_heads: int = None
    rope_theta: float = 10000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            required_keys = {"factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if self.rope_scaling["type"] not in ["linear", "dynamic"]:
                raise ValueError(
                    "rope_scaling 'type' currently only supports 'linear' or 'dynamic"
                )


class DynamicNTKScalingRoPE(nn.Module):
    """Implements the rotary positional encoding with Dynamic NTK scaling."""

    def __init__(
        self,
        dims: int,
        max_position_embeddings: int = 2048,
        traditional: bool = False,
        base: float = 10000,
        scale: float = 1.0,
    ):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.original_base = base
        self.dims = dims
        self.traditional = traditional
        self.scale = scale

    def extra_repr(self):
        return f"{self.dims}, traditional={self.traditional}, max_position_embeddings={self.max_position_embeddings}, scaling_factor={self.scaling_factor}"

    def __call__(self, x, offset: int = 0):
        seq_len = x.shape[1] + offset
        if seq_len > self.max_position_embeddings:
            base = self.original_base * (
                (self.scale * seq_len / self.max_position_embeddings) - (self.scale - 1)
            ) ** (self.dims / (self.dims - 2))
        else:
            base = self.original_base

        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=base,
            scale=self.scale,
            offset=offset,
        )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.n_kv_groups = n_heads // args.num_key_value_heads

        self.head_dim = head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.wqkv = nn.Linear(
            dim, (n_heads + 2 * n_kv_heads) * head_dim, bias=args.bias
        )
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=args.bias)

        rope_scale = (
            1 / args.rope_scaling["factor"]
            if args.rope_scaling is not None and args.rope_scaling["type"] == "linear"
            else 2.0
        )

        self.rope = DynamicNTKScalingRoPE(
            head_dim,
            max_position_embeddings=args.max_position_embeddings,
            traditional=args.rope_traditional,
            base=args.rope_theta,
            scale=rope_scale,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qkv_states = self.wqkv(x)
        qkv_states = qkv_states.reshape(B, L, -1, 2 + self.n_kv_groups, self.head_dim)

        queries = qkv_states[..., : self.n_kv_groups, :]
        queries = queries.reshape(B, L, -1, self.head_dim)
        keys = qkv_states[..., -2, :]
        values = qkv_states[..., -1, :]

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

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
        return self.wo(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = MLP(args.hidden_size, args.intermediate_size)
        self.attention_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out


class InternLM2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size > 0
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.tok_embeddings(inputs)

        mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = InternLM2Model(args)
        if not args.tie_word_embeddings:
            self.output = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.tok_embeddings.as_linear(out)
        else:
            out = self.output(out)
        return out

    def sanitize(self, weights):
        # Remove unused precomputed rotary freqs
        return {k: v for k, v in weights.items() if "attention.rope.inv_freq" not in k}

    @property
    def layers(self):
        return self.model.layers
