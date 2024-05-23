from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    dense_attention_every_n_layers: int
    ff_intermediate_size: int
    gegelu_limit: float
    num_hidden_layers: int
    num_attention_heads: int
    layer_norm_epsilon: float
    vocab_size: int
    num_key_value_heads: int = None
    mup_attn_multiplier: float = 1.0
    mup_use_scaling: bool = True
    mup_embedding_multiplier: float = 10.0
    mup_width_multiplier: float = 8.0
    rope_embedding_base: float = 1000000
    rope_position_scale: float = 1.0


def quick_gelu(x):
    return x * mx.sigmoid(1.702 * x)


def gegelu(x, limit):
    a_gelu, a_linear = x[..., ::2], x[..., 1::2]
    # TODO add limit
    # a_gelu = mx.where(
    #    mx.isinf(a_gelu), a_gelu, a_gelu.clamp(min=None, max=limit)
    # )
    # a_linear = mx.where(
    #    mx.isinf(a_linear), a_linear, a_linear.clamp(min=-limit, max=limit)
    # )
    out_gelu = quick_gelu(a_gelu)
    return out_gelu * (a_linear + 1.0)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.n_q_per_kv = n_heads // n_kv_heads

        self.head_dim = head_dim = args.hidden_size // n_heads

        self.query_key_value = nn.Linear(
            dim, (self.n_heads + 2 * self.n_kv_heads) * head_dim
        )
        self.dense = nn.Linear(dim, dim)

        if args.mup_use_scaling:
            norm_factor = head_dim / args.mup_attn_multiplier
        else:
            norm_factor = math.sqrt(head_dim)
        self.scale = 1.0 / norm_factor

        self.rope = nn.RoPE(
            head_dim,
            traditional=False,
            base=args.rope_embedding_base,
            scale=args.rope_position_scale,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qkv = self.query_key_value(x)
        qkv = qkv.reshape(B, L, -1, self.n_q_per_kv + 2, self.head_dim)
        queries = qkv[..., :-2, :].flatten(-3, -2)
        keys = qkv[..., -2, :]
        values = qkv[..., -1, :]

        # Prepare the queries, keys and values for the attention computation
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

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
    def __init__(self, args):
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.ff_intermediate_size
        self.gegelu_limit = args.gegelu_limit
        self.up_proj = nn.Linear(dim, 2 * hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, dim)

    def __call__(self, x) -> mx.array:
        x = self.up_proj(x)
        return self.down_proj(gegelu(x, self.gegelu_limit))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_epsilon
        )
        self.post_attention_layernorm = nn.LayerNorm(
            args.hidden_size,
            eps=args.layer_norm_epsilon,
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class Phi3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.mup_embedding_multiplier = args.mup_embedding_multiplier
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.final_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_epsilon
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)
        if self.mup_embedding_multiplier:
            h = self.mup_embedding_multiplier * h

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.final_layernorm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.model = Phi3Model(args)
        self.args = args
        self.mup_width_multiplier = args.mup_width_multiplier
        self._dummy_tokenizer_ids = mx.array(
            [100256, 100258, 100259, 100260, 100264, 100265]
            + list(range(100267, 100352))
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        out = self.model.embed_tokens.as_linear(out)
        if self.mup_width_multiplier:
            out = out / self.mup_width_multiplier
        out[self._dummy_tokenizer_ids] = -float("inf")
        return out

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    def sanitize(self, weights):
        # Remove unused precomputed rotary freqs
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
