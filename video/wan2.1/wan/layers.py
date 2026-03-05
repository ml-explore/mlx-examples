# Copyright © 2026 Apple Inc.

"""
Transformer layers for Wan2.1 DiT.

Norms, attention, blocks, and output head. Uses bidirectional (non-causal)
attention with setattr-based block registration for weight remapping
compatibility.
"""

import math
from functools import partial
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .rope import _rope_3d


@partial(mx.compile, shapeless=True)
def _modulate(x, scale, shift):
    return x * (1 + scale) + shift


@partial(mx.compile, shapeless=True)
def _residual_gate(x, y, gate):
    return x.astype(mx.float32) + (y * gate).astype(mx.float32)


_gelu = mx.compile(nn.gelu_approx)


@partial(mx.compile)
def _self_attn_fn(
    x,
    q_w,
    q_b,
    k_w,
    k_b,
    v_w,
    v_b,
    o_w,
    o_b,
    nq_w,
    nk_w,
    n,
    d,
    eps,
    f,
    h,
    w,
    frame_dim,
    height_dim,
    width_dim,
    theta,
):
    B, L, _ = x.shape
    q = mx.matmul(x, q_w.T) + q_b
    k = mx.matmul(x, k_w.T) + k_b
    v = mx.matmul(x, v_w.T) + v_b
    q = mx.fast.rms_norm(q, nq_w, eps)
    k = mx.fast.rms_norm(k, nk_w, eps)
    q = q.reshape(B, L, n, d)
    k = k.reshape(B, L, n, d)
    v = v.reshape(B, L, n, d)
    q = _rope_3d(q, f, h, w, frame_dim, height_dim, width_dim, theta)
    k = _rope_3d(k, f, h, w, frame_dim, height_dim, width_dim, theta)
    scale = d**-0.5
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    x = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    x = x.transpose(0, 2, 1, 3).reshape(B, L, n * d)
    x = mx.matmul(x, o_w.T) + o_b
    return x


@partial(mx.compile)
def _cross_attn_fn(
    x,
    context,
    q_w,
    q_b,
    k_w,
    k_b,
    v_w,
    v_b,
    o_w,
    o_b,
    nq_w,
    nk_w,
    n,
    d,
    eps,
):
    B = x.shape[0]
    L1 = x.shape[1]
    L2 = context.shape[1]
    q = mx.matmul(x, q_w.T) + q_b
    k = mx.matmul(context, k_w.T) + k_b
    v = mx.matmul(context, v_w.T) + v_b
    q = mx.fast.rms_norm(q, nq_w, eps)
    k = mx.fast.rms_norm(k, nk_w, eps)
    q = q.reshape(B, L1, n, d).transpose(0, 2, 1, 3)
    k = k.reshape(B, L2, n, d).transpose(0, 2, 1, 3)
    v = v.reshape(B, L2, n, d).transpose(0, 2, 1, 3)
    x = mx.fast.scaled_dot_product_attention(q, k, v, scale=d**-0.5)
    x = x.transpose(0, 2, 1, 3).reshape(B, L1, n * d)
    x = mx.matmul(x, o_w.T) + o_b
    return x


@partial(mx.compile)
def _cross_attn_mask_fn(
    x,
    context,
    context_lens,
    q_w,
    q_b,
    k_w,
    k_b,
    v_w,
    v_b,
    o_w,
    o_b,
    nq_w,
    nk_w,
    n,
    d,
    eps,
):
    B = x.shape[0]
    L1 = x.shape[1]
    L2 = context.shape[1]
    q = mx.matmul(x, q_w.T) + q_b
    k = mx.matmul(context, k_w.T) + k_b
    v = mx.matmul(context, v_w.T) + v_b
    q = mx.fast.rms_norm(q, nq_w, eps)
    k = mx.fast.rms_norm(k, nk_w, eps)
    q = q.reshape(B, L1, n, d).transpose(0, 2, 1, 3)
    k = k.reshape(B, L2, n, d).transpose(0, 2, 1, 3)
    v = v.reshape(B, L2, n, d).transpose(0, 2, 1, 3)
    positions = mx.arange(L2).reshape(1, 1, 1, L2)
    lengths = context_lens.reshape(-1, 1, 1, 1)
    attn_mask = mx.where(positions >= lengths, float("-inf"), 0.0)
    attn_mask = attn_mask.astype(q.dtype)
    x = mx.fast.scaled_dot_product_attention(q, k, v, scale=d**-0.5, mask=attn_mask)
    x = x.transpose(0, 2, 1, 3).reshape(B, L1, n * d)
    x = mx.matmul(x, o_w.T) + o_b
    return x


@partial(mx.compile, shapeless=True)
def _layer_norm(x, eps):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / mx.sqrt(var + eps)


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class WanLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = mx.ones((dim,))
            self.bias = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        if self.elementwise_affine:
            return mx.fast.layer_norm(x, self.weight, self.bias, self.eps)
        else:
            return _layer_norm(x, self.eps)


class WanSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        if qk_norm:
            self.norm_q = WanRMSNorm(dim, eps=eps)
            self.norm_k = WanRMSNorm(dim, eps=eps)

    def __call__(
        self,
        x: mx.array,
        grid_sizes: list,
        freqs: dict,
    ) -> mx.array:
        f, h, w = grid_sizes[0]
        return _self_attn_fn(
            x,
            self.q.weight,
            self.q.bias,
            self.k.weight,
            self.k.bias,
            self.v.weight,
            self.v.bias,
            self.o.weight,
            self.o.bias,
            self.norm_q.weight,
            self.norm_k.weight,
            self.num_heads,
            self.head_dim,
            self.norm_q.eps,
            f,
            h,
            w,
            freqs["frame"]["full_dim"],
            freqs["height"]["full_dim"],
            freqs["width"]["full_dim"],
            freqs["theta"],
        )


class WanCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        if qk_norm:
            self.norm_q = WanRMSNorm(dim, eps=eps)
            self.norm_k = WanRMSNorm(dim, eps=eps)

    def __call__(
        self,
        x: mx.array,
        context: mx.array,
        context_lens: Optional[mx.array],
    ) -> mx.array:
        if context_lens is not None:
            return self._call_with_mask(x, context, context_lens)
        return _cross_attn_fn(
            x,
            context,
            self.q.weight,
            self.q.bias,
            self.k.weight,
            self.k.bias,
            self.v.weight,
            self.v.bias,
            self.o.weight,
            self.o.bias,
            self.norm_q.weight,
            self.norm_k.weight,
            self.num_heads,
            self.head_dim,
            self.norm_q.eps,
        )

    def _call_with_mask(
        self,
        x: mx.array,
        context: mx.array,
        context_lens,
    ) -> mx.array:
        if not isinstance(context_lens, mx.array):
            context_lens = mx.array(context_lens, dtype=mx.int32)
        return _cross_attn_mask_fn(
            x,
            context,
            context_lens,
            self.q.weight,
            self.q.bias,
            self.k.weight,
            self.k.bias,
            self.v.weight,
            self.v.bias,
            self.o.weight,
            self.o.bias,
            self.norm_q.weight,
            self.norm_k.weight,
            self.num_heads,
            self.head_dim,
            self.norm_q.eps,
        )


class WanAttentionBlock(nn.Module):
    """
    Transformer block with self-attn, cross-attn, and FFN.

    Uses ffn_linear1/ffn_linear2 naming (not nn.Sequential) for weight
    remapping compatibility and selective quantization.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim

        self.norm1 = WanLayerNorm(dim, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        if cross_attn_norm:
            self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True)
        else:
            self.norm3 = None

        self.self_attn = WanSelfAttention(dim, num_heads, qk_norm, eps)
        self.cross_attn = WanCrossAttention(dim, num_heads, qk_norm, eps)

        self.ffn_linear1 = nn.Linear(dim, ffn_dim)
        self.ffn_linear2 = nn.Linear(ffn_dim, dim)

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
        e = (self.modulation + e).astype(mx.float32)
        e = [chunk.squeeze(1) for chunk in e.split(6, axis=1)]

        # Self-attention with modulation
        x_norm = self.norm1(x).astype(mx.float32)
        y = self.self_attn(
            _modulate(x_norm, e[1], e[0]),
            grid_sizes,
            freqs,
        )
        x = _residual_gate(x, y, e[2])

        # Cross-attention
        if self.norm3 is not None:
            x_normed = self.norm3(x)
        else:
            x_normed = x
        x = x + self.cross_attn(x_normed, context, context_lens)

        # FFN with modulation
        x_norm = self.norm2(x).astype(mx.float32)
        y = self.ffn_linear2(_gelu(self.ffn_linear1(_modulate(x_norm, e[4], e[3]))))
        x = _residual_gate(x, y, e[5])

        return x


class Head(nn.Module):
    """Output head with modulation. Uses raw weight arrays for remapping compat."""

    def __init__(
        self,
        dim: int,
        out_dim: int,
        patch_size: Tuple[int, int, int],
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        out_features = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        scale = 1.0 / dim**0.5
        self.head_weight = mx.random.uniform(
            low=-scale, high=scale, shape=(out_features, dim)
        )
        self.head_bias = mx.zeros((out_features,))
        self.modulation = mx.zeros((1, 2, dim))

    def __call__(self, x: mx.array, e: mx.array) -> mx.array:
        e = (self.modulation + e[:, None, :]).astype(mx.float32)
        e = e.split(2, axis=1)
        x = x.astype(mx.float32)
        x_norm = self.norm(x).astype(mx.float32)
        x = (
            mx.matmul(
                _modulate(x_norm, e[1].squeeze(1), e[0].squeeze(1)),
                self.head_weight.T,
            )
            + self.head_bias
        )
        return x
