"""
TwoStreamInterleaveTransformer, the main triplane<->image cross-attention
backbone. Ported from SF3D's `sf3d/models/transformers/backbone.py` (only
the classes actually reachable from this checkpoint's backbone_cls -
TriplaneAttention/SingleStreamTransformer are unused and skipped).

All dims are 1024 (triplane_channels == raw_triplane_channels ==
raw_image_channels == latent_dim, num_attention_heads=16 * attention_head_dim
=64 = 1024) in this checkpoint - no need to handle the general case.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

DIM = 1024
NUM_HEADS = 16
HEAD_DIM = DIM // NUM_HEADS
NUM_BLOCKS = 4
NUM_BASIC_BLOCKS = 3
NUM_LATENTS = 1792
NORM_GROUPS = 32
GN_EPS = 1e-6
LN_EPS = 1e-5  # nn.LayerNorm default eps in the source (not overridden here)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, kv_dim: int):
        super().__init__()
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(kv_dim, dim, bias=False)
        self.wv = nn.Linear(kv_dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.scale = HEAD_DIM**-0.5

    def __call__(self, x_q: mx.array, x_kv: mx.array) -> mx.array:
        b, nq, c = x_q.shape
        nkv = x_kv.shape[1]
        q = self.wq(x_q).reshape(b, nq, NUM_HEADS, c // NUM_HEADS).transpose(0, 2, 1, 3)
        k = (
            self.wk(x_kv)
            .reshape(b, nkv, NUM_HEADS, c // NUM_HEADS)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.wv(x_kv)
            .reshape(b, nkv, NUM_HEADS, c // NUM_HEADS)
            .transpose(0, 2, 1, 3)
        )
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(b, nq, c)
        return self.proj(out)


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def __call__(self, x: mx.array) -> mx.array:
        x, gate = mx.split(self.proj(x), 2, axis=-1)
        return x * nn.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner_dim = dim * mult
        self.geglu = GEGLU(dim, inner_dim)
        self.out = nn.Linear(inner_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out(self.geglu(x))


class BasicBlock(nn.Module):
    """Self-attn (attn1) + cross-attn to a fixed side input (attn2) + FF."""

    def __init__(self, dim: int, kv_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=LN_EPS)
        self.attn1 = CrossAttention(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=LN_EPS)
        self.attn2 = CrossAttention(dim, kv_dim)
        self.norm3 = nn.LayerNorm(dim, eps=LN_EPS)
        self.ff = FeedForward(dim)

    def __call__(self, z: mx.array, x: mx.array) -> mx.array:
        zn = self.norm1(z)
        z = z + self.attn1(zn, zn)
        zn = self.norm2(z)
        z = z + self.attn2(zn, x)
        zn = self.norm3(z)
        z = z + self.ff(zn)
        return z


class FuseBlock(nn.Module):
    """Fuse x into z with cross attention. norm_x_input=False in this
    checkpoint (config.norm_x_input: false) - x is used raw, unnormalized,
    and there is no norm_x weight in the checkpoint (confirmed absent)."""

    def __init__(self, dim_z: int, dim_x: int):
        super().__init__()
        self.attn = CrossAttention(dim_z, dim_x)
        self.norm_z1 = nn.LayerNorm(dim_z, eps=LN_EPS)
        self.norm_z2 = nn.LayerNorm(dim_z, eps=LN_EPS)
        self.ff = FeedForward(dim_z)

    def __call__(self, z: mx.array, x: mx.array) -> mx.array:
        z = z + self.attn(self.norm_z1(z), x)
        z = z + self.ff(self.norm_z2(z))
        return z


class TwoStreamBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuse_block_in = FuseBlock(DIM, DIM)
        self.transformer_block = [BasicBlock(DIM, DIM) for _ in range(NUM_BASIC_BLOCKS)]
        self.fuse_block_out = FuseBlock(DIM, DIM)

    def __call__(self, latent: mx.array, triplane: mx.array, cross_input: mx.array):
        latent = self.fuse_block_in(latent, triplane)
        for block in self.transformer_block:
            latent = block(latent, cross_input)
        triplane = self.fuse_block_out(triplane, latent)
        return latent, triplane


class TwoStreamInterleaveTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_triplane = nn.GroupNorm(
            NORM_GROUPS, DIM, eps=GN_EPS, pytorch_compatible=True
        )
        self.proj_triplane = nn.Linear(DIM, DIM)
        self.norm_image = nn.LayerNorm(DIM, eps=LN_EPS)
        self.proj_image = nn.Linear(DIM, DIM)
        self.norm_latent = nn.LayerNorm(DIM, eps=LN_EPS)
        self.proj_latent = nn.Linear(DIM, DIM)
        self.latent_init = mx.zeros((1, NUM_LATENTS, DIM))
        self.main_blocks = [TwoStreamBlock() for _ in range(NUM_BLOCKS)]
        self.proj_out = nn.Linear(DIM, DIM)

    def __call__(
        self, hidden_states: mx.array, encoder_hidden_states: mx.array
    ) -> mx.array:
        # hidden_states: [B, DIM, N_triplane] (channel-first, like torch)
        # encoder_hidden_states: [B, N_image, DIM] (already channel-last)
        b = hidden_states.shape[0]

        triplane_tokens = hidden_states.transpose(0, 2, 1)  # [B, N, DIM]
        triplane_tokens = self.norm_triplane(
            triplane_tokens
        )  # GroupNorm, feature-dim-last
        triplane_tokens = self.proj_triplane(triplane_tokens)

        image_tokens = self.norm_image(encoder_hidden_states)
        image_tokens = self.proj_image(image_tokens)

        init_latents = mx.broadcast_to(self.latent_init, (b, NUM_LATENTS, DIM))
        init_latents = self.norm_latent(init_latents)
        init_latents = self.proj_latent(init_latents)

        latent_tokens = mx.concatenate([image_tokens, init_latents], axis=1)

        for block in self.main_blocks:
            latent_tokens, triplane_tokens = block(
                latent_tokens, triplane_tokens, encoder_hidden_states
            )

        triplane_tokens = self.proj_out(triplane_tokens).transpose(
            0, 2, 1
        )  # back to [B, DIM, N]
        triplane_tokens = triplane_tokens + hidden_states
        return triplane_tokens


def load_into(
    model: TwoStreamInterleaveTransformer, weights: dict, prefix: str = "backbone."
):
    model.norm_triplane.weight = weights[prefix + "norm_triplane.weight"]
    model.norm_triplane.bias = weights[prefix + "norm_triplane.bias"]
    model.proj_triplane.weight = weights[prefix + "proj_triplane.weight"]
    model.proj_triplane.bias = weights[prefix + "proj_triplane.bias"]
    model.norm_image.weight = weights[prefix + "norm_image.weight"]
    model.norm_image.bias = weights[prefix + "norm_image.bias"]
    model.proj_image.weight = weights[prefix + "proj_image.weight"]
    model.proj_image.bias = weights[prefix + "proj_image.bias"]
    model.norm_latent.weight = weights[prefix + "norm_latent.weight"]
    model.norm_latent.bias = weights[prefix + "norm_latent.bias"]
    model.proj_latent.weight = weights[prefix + "proj_latent.weight"]
    model.proj_latent.bias = weights[prefix + "proj_latent.bias"]
    model.latent_init = weights[prefix + "latent_init"]
    model.proj_out.weight = weights[prefix + "proj_out.weight"]
    model.proj_out.bias = weights[prefix + "proj_out.bias"]

    def load_cross_attn(attn: CrossAttention, p: str):
        attn.wq.weight = weights[p + "wq.weight"]
        attn.wk.weight = weights[p + "wk.weight"]
        attn.wv.weight = weights[p + "wv.weight"]
        attn.proj.weight = weights[p + "proj.weight"]
        attn.proj.bias = weights[p + "proj.bias"]

    def load_ff(ff: FeedForward, p: str):
        ff.geglu.proj.weight = weights[p + "net.0.proj.weight"]
        ff.geglu.proj.bias = weights[p + "net.0.proj.bias"]
        ff.out.weight = weights[p + "net.2.weight"]
        ff.out.bias = weights[p + "net.2.bias"]

    def load_fuse_block(fb: FuseBlock, p: str):
        load_cross_attn(fb.attn, p + "attn.")
        fb.norm_z1.weight = weights[p + "norm_z1.weight"]
        fb.norm_z1.bias = weights[p + "norm_z1.bias"]
        fb.norm_z2.weight = weights[p + "norm_z2.weight"]
        fb.norm_z2.bias = weights[p + "norm_z2.bias"]
        load_ff(fb.ff, p + "ff.")

    for bi, block in enumerate(model.main_blocks):
        bp = f"{prefix}main_blocks.{bi}."
        load_fuse_block(block.fuse_block_in, bp + "fuse_block_in.")
        load_fuse_block(block.fuse_block_out, bp + "fuse_block_out.")
        for ti, tb in enumerate(block.transformer_block):
            tp = f"{bp}transformer_block.{ti}."
            tb.norm1.weight = weights[tp + "norm1.weight"]
            tb.norm1.bias = weights[tp + "norm1.bias"]
            load_cross_attn(tb.attn1, tp + "attn1.")
            tb.norm2.weight = weights[tp + "norm2.weight"]
            tb.norm2.bias = weights[tp + "norm2.bias"]
            load_cross_attn(tb.attn2, tp + "attn2.")
            tb.norm3.weight = weights[tp + "norm3.weight"]
            tb.norm3.bias = weights[tp + "norm3.bias"]
            load_ff(tb.ff, tp + "ff.")
