# Copyright © 2026 Apple Inc.

"""
Transformer layers for Wan2.1 DiT.

Norms, attention, blocks, and output head. Uses bidirectional (non-causal)
attention with fused norm+modulate via mx.fast.layer_norm.
"""

import math
from functools import partial
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .rope import rope_apply


@partial(mx.compile, shapeless=True)
def _residual_gate(x, y, gate):
    return x + y * gate


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class WanSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.o = nn.Linear(dim, dim)

        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

    def _attend(self, x, grid_sizes, freqs):
        """Compute self-attention. Returns attn output [B, n, L, d]."""
        B, L, _ = x.shape
        n, d = self.num_heads, self.head_dim

        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        q = self.norm_q(q)
        k = self.norm_k(k)

        q = q.reshape(B, L, n, d)
        k = k.reshape(B, L, n, d)
        v = v.reshape(B, L, n, d)

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5)

    def __call__(self, x, grid_sizes, freqs):
        B, L, C = x.shape
        attn = self._attend(x, grid_sizes, freqs)
        return self.o(attn.transpose(0, 2, 1, 3).reshape(B, L, C))


class WanCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.o = nn.Linear(dim, dim)

        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

    def _attend(self, x, context, context_lens):
        """Compute text cross-attention. Returns (q, attn_out) both [B, n, L, d]."""
        B = x.shape[0]
        L1, L2 = x.shape[1], context.shape[1]
        n, d = self.num_heads, self.head_dim

        q = self.norm_q(self.q(x))
        kv = self.kv(context)
        k, v = mx.split(kv, 2, axis=-1)
        k = self.norm_k(k)

        q = q.reshape(B, L1, n, d).transpose(0, 2, 1, 3)
        k = k.reshape(B, L2, n, d).transpose(0, 2, 1, 3)
        v = v.reshape(B, L2, n, d).transpose(0, 2, 1, 3)

        if context_lens is not None:
            if not isinstance(context_lens, mx.array):
                context_lens = mx.array(context_lens, dtype=mx.int32)
            positions = mx.arange(L2).reshape(1, 1, 1, L2)
            lengths = context_lens.reshape(-1, 1, 1, 1)
            attn_mask = mx.where(positions >= lengths, float("-inf"), 0.0)
            attn_mask = attn_mask.astype(q.dtype)
            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=d**-0.5, mask=attn_mask
            )
        else:
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=d**-0.5)

        return q, out

    def __call__(self, x, context, context_lens):
        _, attn = self._attend(x, context, context_lens)
        B, _, L1, _ = attn.shape
        x = attn.transpose(0, 2, 1, 3).reshape(B, L1, self.dim)
        return self.o(x)


T5_CONTEXT_TOKEN_NUMBER = 512


class WanI2VCrossAttention(WanCrossAttention):
    """Cross-attention with separate image and text paths for I2V."""

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__(dim, num_heads, eps)
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps)

    def __call__(self, x, context, context_lens):
        img_ctx_len = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :img_ctx_len]
        context_txt = context[:, img_ctx_len:]

        # Text attention
        q, x_txt = self._attend(x, context_txt, context_lens)

        # Image attention (no mask, reuses q)
        B, L1 = x.shape[:2]
        n, d = self.num_heads, self.head_dim
        L_img = context_img.shape[1]
        ki = self.norm_k_img(self.k_img(context_img))
        vi = self.v_img(context_img)
        ki = ki.reshape(B, L_img, n, d).transpose(0, 2, 1, 3)
        vi = vi.reshape(B, L_img, n, d).transpose(0, 2, 1, 3)
        x_img = mx.fast.scaled_dot_product_attention(q, ki, vi, scale=d**-0.5)

        x = (x_txt + x_img).transpose(0, 2, 1, 3).reshape(B, L1, self.dim)
        return self.o(x)


_cross_attn_classes = {
    "t2v": WanCrossAttention,
    "i2v": WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):
    """
    Transformer block with self-attn, cross-attn, and FFN.

    Uses fused norm+modulate via mx.fast.layer_norm where the modulation
    scale/shift are passed as weight/bias. Requires sanitize to bake 1+
    into modulation scale positions.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        cross_attn_type: str = "t2v",
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps

        if cross_attn_norm:
            self.norm3 = nn.LayerNorm(dim, eps=eps)
        else:
            self.norm3 = None

        self.self_attn = WanSelfAttention(dim, num_heads, eps)
        self.cross_attn = _cross_attn_classes[cross_attn_type](dim, num_heads, eps)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approx="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        self.modulation = mx.zeros((1, 6, dim))

    def __call__(
        self,
        x: mx.array,
        e: mx.array,
        grid_sizes: list,
        freqs: dict,
        context: mx.array,
        context_lens: Optional[mx.array],
    ) -> mx.array:
        e = self.modulation + e

        # Self-attention: fused norm + modulate
        y = self.self_attn(
            mx.fast.layer_norm(x, e[0, 1], e[0, 0], self.eps),
            grid_sizes,
            freqs,
        )
        x = _residual_gate(x, y, e[:, 2])

        # Cross-attention
        if self.norm3 is not None:
            x_normed = self.norm3(x)
        else:
            x_normed = x
        x = x + self.cross_attn(x_normed, context, context_lens)

        # FFN: fused norm + modulate
        y = self.ffn(mx.fast.layer_norm(x, e[0, 4], e[0, 3], self.eps))
        x = _residual_gate(x, y, e[:, 5])

        return x


class Head(nn.Module):
    """Output head with fused norm+modulate and nn.Linear."""

    def __init__(
        self,
        dim: int,
        out_dim: int,
        patch_size: Tuple[int, int, int],
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        out_features = math.prod(patch_size) * out_dim
        self.linear = nn.Linear(dim, out_features)
        self.modulation = mx.zeros((1, 2, dim))

    def __call__(self, x: mx.array, e: mx.array) -> mx.array:
        e = self.modulation + e[:, None, :]
        x = mx.fast.layer_norm(x, e[0, 1], e[0, 0], self.eps)
        return self.linear(x)
