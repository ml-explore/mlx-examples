# Copyright © 2026 Apple Inc.

"""
T5 text encoder for Wan2.1.

UMT5-XXL encoder (4096 dim, 24 layers, 64 heads) with gated GELU FFN
and per-layer relative position embeddings.
"""

import math
import re
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
from einops import rearrange


class T5RelativeEmbedding(nn.Module):
    def __init__(self, num_buckets, num_heads, bidirectional=True, max_dist=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, rel_pos):
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).astype(mx.int32) * num_buckets
            rel_pos = mx.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = mx.zeros_like(rel_pos, dtype=mx.int32)
            rel_pos = -mx.minimum(rel_pos, mx.zeros_like(rel_pos))

        max_exact = num_buckets // 2
        is_small = rel_pos < max_exact
        scale = (num_buckets - max_exact) / math.log(self.max_dist / max_exact)
        rel_pos_large = max_exact + (
            mx.log(rel_pos.astype(mx.float32) / max_exact) * scale
        ).astype(mx.int32)
        rel_pos_large = mx.minimum(rel_pos_large, num_buckets - 1)
        rel_buckets = rel_buckets + mx.where(is_small, rel_pos, rel_pos_large)
        return rel_buckets

    def __call__(self, lq, lk):
        query_pos = mx.arange(lq)[:, None]
        key_pos = mx.arange(lk)[None, :]
        rel_pos = key_pos - query_pos
        rel_buckets = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_buckets)
        return rel_pos_embeds.transpose(2, 0, 1)[None, :, :, :]


class T5Attention(nn.Module):
    def __init__(self, dim, dim_attn, num_heads):
        super().__init__()
        assert dim_attn % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)

    def __call__(self, x, context=None, mask=None, pos_bias=None):
        context = x if context is None else context
        b = x.shape[0]
        n, c = self.num_heads, self.head_dim

        q = rearrange(self.q(x), "b s (n c) -> b n s c", n=n)
        k = rearrange(self.k(context), "b s (n c) -> b n s c", n=n)
        v = rearrange(self.v(context), "b s (n c) -> b n s c", n=n)

        attn_bias = mx.zeros((b, n, q.shape[2], k.shape[2]), dtype=x.dtype)
        if pos_bias is not None:
            attn_bias = attn_bias + pos_bias
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, None, None, :]
            else:
                mask = mask[:, None, :, :]
            attn_bias = mx.where(mask == 0, -1e9, attn_bias)

        # T5 does NOT use sqrt(d) scaling
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=attn_bias)
        out = rearrange(out, "b n s c -> b s (n c)")
        return self.o(out)


class T5FeedForward(nn.Module):
    def __init__(self, dim, dim_ffn):
        super().__init__()
        self.gate = nn.Linear(dim, dim_ffn, bias=False)
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)

    def __call__(self, x):
        return self.fc2(self.fc1(x) * nn.gelu_approx(self.gate(x)))


class T5SelfAttention(nn.Module):
    def __init__(self, dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos=True):
        super().__init__()
        self.shared_pos = shared_pos
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = T5Attention(dim, dim_attn, num_heads)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.ffn = T5FeedForward(dim, dim_ffn)
        self.pos_embedding = (
            None
            if shared_pos
            else T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)
        )

    def __call__(self, x, mask=None, pos_bias=None):
        e = pos_bias if self.shared_pos else self.pos_embedding(x.shape[1], x.shape[1])
        x = x + self.attn(self.norm1(x), mask=mask, pos_bias=e)
        x = x + self.ffn(self.norm2(x))
        return x


class T5Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim,
        dim_attn,
        dim_ffn,
        num_heads,
        num_layers,
        num_buckets,
        shared_pos=True,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.shared_pos = shared_pos
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = (
            T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)
            if shared_pos
            else None
        )
        self.blocks = [
            T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos)
            for _ in range(num_layers)
        ]
        self.norm = nn.RMSNorm(dim, eps=1e-6)

    def __call__(self, ids, mask=None):
        x = self.token_embedding(ids)
        seq_len = x.shape[1]
        e = self.pos_embedding(seq_len, seq_len) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask=mask, pos_bias=e)
        x = self.norm(x)
        return x

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Remap PyTorch T5 keys to MLX format."""
        remapped = {}
        for key, value in weights.items():
            new_key = key
            if new_key.startswith("model."):
                new_key = new_key[6:]
            if "ffn.gate.1" in new_key:
                continue
            if "dropout" in new_key:
                continue
            new_key = re.sub(r"ffn\.gate\.0\.", "ffn.gate.", new_key)
            remapped[new_key] = value
        return remapped


def create_umt5_xxl_encoder() -> T5Encoder:
    return T5Encoder(
        vocab_size=256384,
        dim=4096,
        dim_attn=4096,
        dim_ffn=10240,
        num_heads=64,
        num_layers=24,
        num_buckets=32,
        shared_pos=False,
    )
