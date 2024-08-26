# Copyright © 2023-2024 Apple Inc.

import math
from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, KVCache, create_attention_mask


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
    num_key_value_heads: Optional[int] = None
    mup_attn_multiplier: float = 1.0
    mup_use_scaling: bool = True
    mup_embedding_multiplier: float = 10.0
    mup_width_multiplier: float = 8.0
    rope_embedding_base: float = 1000000
    rope_position_scale: float = 1.0
    blocksparse_block_size: Tuple[int] = (64,)
    blocksparse_num_local_blocks: int = 16
    blocksparse_vert_stride: int = 8


@partial(mx.compile, shapeless=True)
def gegelu_impl(a_gelu, a_linear, limit):
    a_gelu = mx.where(
        mx.isinf(a_gelu),
        a_gelu,
        mx.clip(a_gelu, a_min=None, a_max=limit),
    )
    a_linear = mx.where(
        mx.isinf(a_linear),
        a_linear,
        mx.clip(a_linear, a_min=-limit, a_max=limit),
    )
    out_gelu = a_gelu * mx.sigmoid(1.702 * a_gelu)
    return out_gelu * (a_linear + 1.0)


def gegelu(x, limit):
    a_gelu, a_linear = x[..., ::2], x[..., 1::2]
    return gegelu_impl(a_gelu, a_linear, limit)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
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

        if layer_idx % args.dense_attention_every_n_layers == 0:
            self.block_sparse = True
            self.blocksparse_block_size = args.blocksparse_block_size
            if self.blocksparse_block_size not in (32, 64):
                raise ValueError(
                    f"Unsupported block size {self.blocksparse_block_size}"
                )
            self.blocksparse_num_local_blocks = args.blocksparse_num_local_blocks
            self.blocksparse_vert_stride = args.blocksparse_vert_stride
        else:
            self.block_sparse = False

    def _block_sparse_mask(self, q_len, kv_len):
        vert_stride = self.blocksparse_vert_stride
        local_blocks = self.blocksparse_num_local_blocks
        block_size = self.blocksparse_block_size
        n_heads = self.n_heads

        kv_blocks = (kv_len + block_size - 1) // block_size
        q_blocks = (q_len + block_size - 1) // block_size
        q_pos = mx.arange(kv_blocks - q_blocks, kv_blocks)[None, :, None]
        k_pos = mx.arange(kv_blocks)[None, None]

        mask_vert_strided = (
            mx.arange(kv_blocks)[None, :] + mx.arange(1, n_heads + 1)[:, None]
        ) % vert_stride
        mask_vert_strided = (mask_vert_strided == 0)[:, None, :]

        block_mask = (q_pos >= k_pos) & (
            (q_pos - k_pos < local_blocks) | mask_vert_strided
        )
        block_mask = block_mask.reshape(
            self.n_kv_heads, self.n_q_per_kv, *block_mask.shape[-2:]
        )
        dense_mask = mx.repeat(
            mx.repeat(block_mask, block_size, axis=-1), block_size, axis=-2
        )
        return block_mask, dense_mask[..., -q_len:, :kv_len]

    def _block_sparse_attention(self, queries, keys, values, scale, mask):
        queries = scale * queries
        B = queries.shape[0]
        L = queries.shape[2]
        queries = mx.reshape(queries, (B, self.n_kv_heads, self.n_q_per_kv, L, -1))
        keys = mx.expand_dims(keys, 2)
        values = mx.expand_dims(values, 2)

        # TODO get rid of dense mask if we have a fill value
        block_mask, dense_mask = self._block_sparse_mask(L, keys.shape[-2])
        scores = queries @ mx.swapaxes(keys, -1, -2)
        # TODO, uncomment when faster
        # scores = mx.block_masked_mm(
        #   queries,
        #   mx.swapaxes(keys, -1, -2),
        #   mask_out=block_mask,
        #   block_size=self.blocksparse_block_size,
        # )

        if mask is not None:
            scores = scores + mask
        scores = scores + mx.where(
            dense_mask, mx.array(0, scores.dtype), mx.array(-float("inf"), scores.dtype)
        )
        scores = mx.softmax(scores, axis=-1, precise=True)

        output = scores @ values
        # TODO, uncomment when faster
        # output = mx.block_masked_mm(
        #    scores, values, mask_lhs=block_mask, block_size=self.blocksparse_block_size
        # )
        return mx.reshape(output, (B, self.n_heads, L, -1))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
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

        if self.block_sparse:
            output = self._block_sparse_attention(
                queries, keys, values, scale=self.scale, mask=mask
            )
        else:
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
    def __init__(self, args: ModelArgs, layer_idx):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args, layer_idx)
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
        cache: Optional[KVCache] = None,
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
            TransformerBlock(args=args, layer_idx=l)
            for l in range(args.num_hidden_layers)
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

        mask = create_attention_mask(h, cache)

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
