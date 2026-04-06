# Copyright © 2026 Apple Inc.

"""
Wan2.1 bidirectional DiT (Diffusion Transformer) for video generation.

Supports 1.3B and 14B model sizes with text-to-video (t2v) and
image-to-video (i2v) modes. Uses bidirectional attention with
nn.Sequential embeddings and list-based block storage.
"""

import math
import re
from functools import partial
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from einops import rearrange

from .layers import Head, WanAttentionBlock


# shapeless=True: avoids recompilation across varying input shapes.
@partial(mx.compile, shapeless=True)
def sinusoidal_embedding_1d(dim: int, position: mx.array) -> mx.array:
    assert dim % 2 == 0
    half = dim // 2
    dtype = position.dtype
    position = position.astype(mx.float32)
    sinusoid = (
        position[:, None]
        * mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)[None, :]
    )
    return mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=1).astype(dtype)


class WanModel(nn.Module):
    def __init__(
        self,
        model_type: str = "t2v",
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.freq_dim = freq_dim

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size, bias=True
        )

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approx="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

        # Image embedding MLP for I2V: LayerNorm -> Linear -> GELU -> Linear -> LayerNorm
        if model_type == "i2v":
            clip_dim = 1280
            self.img_emb_norm1 = nn.LayerNorm(clip_dim)
            self.img_emb_linear1 = nn.Linear(clip_dim, clip_dim)
            self.img_emb_linear2 = nn.Linear(clip_dim, dim)
            self.img_emb_norm2 = nn.LayerNorm(dim)

        # Transformer blocks as list
        self.blocks = [
            WanAttentionBlock(
                dim,
                ffn_dim,
                num_heads,
                cross_attn_norm,
                eps,
                cross_attn_type=model_type,
            )
            for _ in range(num_layers)
        ]

        # Output head
        self.head = Head(dim, out_dim, patch_size, eps)

    def _embed_image(self, clip_fea: mx.array) -> mx.array:
        """Project CLIP features through img_emb MLP."""
        x = self.img_emb_norm1(clip_fea)
        x = self.img_emb_linear1(x)
        x = nn.gelu(x)
        x = self.img_emb_linear2(x)
        x = self.img_emb_norm2(x)
        return x

    def compute_time_embedding(self, t: mx.array):
        """Compute time embeddings for TeaCache. Returns (t_emb, e0).
        t_emb: [1, dim] (pre-projection, used by head)
        e0: [1, 6*dim] (projected, used for block modulation)"""
        e = sinusoidal_embedding_1d(self.freq_dim, t)
        t_emb = self.time_embedding(e)
        e0 = self.time_projection(t_emb)
        return t_emb, e0

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        context: mx.array,
        block_residual: Optional[mx.array] = None,
        precomputed_time: Optional[Tuple[mx.array, mx.array]] = None,
        clip_fea: Optional[mx.array] = None,
        first_frame: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Forward pass for t2v and i2v.

        Args:
            x: Input latent [F, H, W, C_in] (channels-last)
            t: Timestep [1]
            context: Text embedding [L, C_text]
            block_residual: Precomputed block residual for TeaCache skip
            precomputed_time: (t_emb, e0) tuple for TeaCache
            clip_fea: CLIP image features [1, 257, 1280] (I2V only)
            first_frame: Image conditioning [F, H, W, C_cond] (I2V only).
               Concatenated channel-wise with x before patchify (in_dim=36).

        Returns:
            (output, block_residual): output latent [F, H, W, C_out] and
            block residual for TeaCache caching (None-equivalent zeros when
            using cached residual).
        """
        # Channel-concat image conditioning before patchify (I2V)
        if first_frame is not None:
            x = mx.concatenate([x, first_frame], axis=-1)

        # Patchify: [F, H, W, C] -> [1, F, H, W, C] -> conv3d -> [1, Fp, Hp, Wp, dim]
        x = self.patch_embedding(x[None])
        _, Fp, Hp, Wp, _ = x.shape
        grid_sizes = [[Fp, Hp, Wp]]
        x = x.reshape(1, Fp * Hp * Wp, self.dim)

        # Embed context: [L, C_text] -> [1, text_len, dim]
        context = self.text_embedding(context[None])

        # Prepend projected CLIP features to context (I2V)
        if clip_fea is not None:
            clip_proj = self._embed_image(clip_fea)
            context = mx.concatenate([clip_proj, context], axis=1)

        # Time embedding
        if precomputed_time is not None:
            t_emb, e = precomputed_time[0], precomputed_time[1]
        else:
            e = sinusoidal_embedding_1d(self.freq_dim, t)
            t_emb = self.time_embedding(e)
            e = self.time_projection(t_emb)
        e = e.reshape(1, 6, self.dim)

        # Transformer blocks
        if block_residual is not None:
            x = x + block_residual
            new_residual = block_residual  # pass through (caller won't cache this)
        else:
            x_in = x
            for block in self.blocks:
                x = block(x, e, grid_sizes, context)
            new_residual = x - x_in

        # Output head
        x = self.head(x, t_emb)

        # Unpatchify: [1, seq_len, patch_features] -> [F, H, W, C]
        pt, ph, pw = self.patch_size
        output = rearrange(
            x[0],
            "(Fp Hp Wp) (pt ph pw c) -> (Fp pt) (Hp ph) (Wp pw) c",
            Fp=Fp,
            Hp=Hp,
            Wp=Wp,
            pt=pt,
            ph=ph,
            pw=pw,
        )
        return output, new_residual

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Remap PyTorch checkpoint keys to MLX model format."""
        remapped = {}
        for key, value in weights.items():
            new_key = key

            # Skip fp8 scale metadata from LightX2V quantized checkpoints
            if "weight_scale" in new_key:
                continue

            # Remove model. prefix
            if new_key.startswith("model."):
                new_key = new_key[6:]

            # PyTorch Conv3d [O,I,kT,kH,kW] -> MLX Conv3d [O,kT,kH,kW,I]
            if (
                "patch_embedding" in new_key
                and "weight" in new_key
                and len(value.shape) == 5
            ):
                value = mx.transpose(value, (0, 2, 3, 4, 1))

            # PyTorch nn.Sequential uses flat keys ("ffn.0."), MLX nests under ".layers." ("ffn.layers.0.")
            new_key = new_key.replace("ffn.0.", "ffn.layers.0.")
            new_key = new_key.replace("ffn.2.", "ffn.layers.2.")

            new_key = new_key.replace("text_embedding.0.", "text_embedding.layers.0.")
            new_key = new_key.replace("text_embedding.2.", "text_embedding.layers.2.")

            new_key = new_key.replace("time_embedding.0.", "time_embedding.layers.0.")
            new_key = new_key.replace("time_embedding.2.", "time_embedding.layers.2.")

            new_key = new_key.replace("time_projection.1.", "time_projection.layers.1.")

            # head.head -> head.linear
            new_key = new_key.replace("head.head.", "head.linear.")

            # img_emb.proj.N -> img_emb_* (I2V MLPProj)
            new_key = re.sub(r"img_emb\.proj\.0\.(\w+)", r"img_emb_norm1.\1", new_key)
            new_key = re.sub(r"img_emb\.proj\.1\.(\w+)", r"img_emb_linear1.\1", new_key)
            new_key = re.sub(r"img_emb\.proj\.3\.(\w+)", r"img_emb_linear2.\1", new_key)
            new_key = re.sub(r"img_emb\.proj\.4\.(\w+)", r"img_emb_norm2.\1", new_key)

            remapped[new_key] = value

        # Merge separate Q/K/V into QKV for self-attention,
        # and K/V into KV for cross-attention
        remapped = WanModel._merge_qkv_weights(remapped)

        # Modulation vectors are [shift, scale, gate, ...]. The DiT block applies
        # them as x * (1 + scale) + shift, but we fuse the "1 +" into the stored
        # scale weights here so the forward pass is just x * scale + shift.
        for key in list(remapped.keys()):
            if key.endswith(".modulation"):
                v = remapped[key]
                if v.shape[1] == 6:  # block modulation [1, 6, dim]
                    # Add 1 to scale positions (1 and 4)
                    remapped[key] = v + mx.array([0, 1, 0, 0, 1, 0])[:, None]
                elif v.shape[1] == 2:  # head modulation [1, 2, dim]
                    remapped[key] = v + mx.array([0, 1])[:, None]

        return remapped

    @staticmethod
    def _merge_qkv_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Merge separate q/k/v weights into qkv (self-attn) and kv (cross-attn)."""
        merged = {}
        consumed = set()

        for key in weights:
            # Self-attention: merge q, k, v -> qkv
            m = re.match(r"(blocks\.\d+\.self_attn)\.(q)\.(weight|bias)$", key)
            if m:
                prefix, _, param = m.groups()
                q_key = f"{prefix}.q.{param}"
                k_key = f"{prefix}.k.{param}"
                v_key = f"{prefix}.v.{param}"
                if q_key in weights and k_key in weights and v_key in weights:
                    merged[f"{prefix}.qkv.{param}"] = mx.concatenate(
                        [weights[q_key], weights[k_key], weights[v_key]], axis=0
                    )
                    consumed.update([q_key, k_key, v_key])
                continue

            # Cross-attention: merge k, v -> kv (q stays separate)
            m = re.match(r"(blocks\.\d+\.cross_attn)\.(k)\.(weight|bias)$", key)
            if m:
                prefix, _, param = m.groups()
                k_key = f"{prefix}.k.{param}"
                v_key = f"{prefix}.v.{param}"
                if k_key in weights and v_key in weights:
                    merged[f"{prefix}.kv.{param}"] = mx.concatenate(
                        [weights[k_key], weights[v_key]], axis=0
                    )
                    consumed.update([k_key, v_key])
                continue

        # Copy all non-consumed keys
        for key, value in weights.items():
            if key not in consumed:
                merged[key] = value

        return merged
