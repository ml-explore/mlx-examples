# Copyright © 2026 Apple Inc.

"""
Wan2.1 bidirectional DiT (Diffusion Transformer) for video generation.

Supports 1.3B and 14B model sizes with text-to-video (t2v) and
image-to-video (i2v) modes. Uses bidirectional attention
with setattr-based block registration for weight remapping compatibility.
"""

import math
import re
from functools import partial
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from einops import rearrange

from .layers import Head, WanAttentionBlock
from .rope import precompute_rope_freqs


@partial(mx.compile, shapeless=True)
def sinusoidal_embedding_1d(dim: int, position: mx.array) -> mx.array:
    assert dim % 2 == 0
    half = dim // 2
    position = position.astype(mx.float32)
    sinusoid = (
        position[:, None]
        * mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)[None, :]
    )
    return mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=1)


@partial(mx.compile, shapeless=True)
def _embed_text_fn(context, w1, b1, w2, b2):
    x = mx.matmul(context, w1.T) + b1
    x = nn.gelu_approx(x)
    x = mx.matmul(x, w2.T) + b2
    return x


@partial(mx.compile, shapeless=True)
def _embed_time_fn(freq_dim, t, w1, b1, w2, b2):
    x = sinusoidal_embedding_1d(freq_dim, t)
    x = mx.matmul(x, w1.T) + b1
    x = nn.silu(x)
    x = mx.matmul(x, w2.T) + b2
    return x


@partial(mx.compile, shapeless=True)
def _project_time_fn(e, w, b):
    x = nn.silu(e)
    x = mx.matmul(x, w.T) + b
    return x


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
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = dim // num_heads

        # Patch embedding: raw Conv3d weight/bias
        self.patch_embedding_weight = mx.random.normal((dim, *patch_size, in_dim)) * (
            1.0 / (in_dim * math.prod(patch_size)) ** 0.5
        )
        self.patch_embedding_bias = mx.zeros((dim,))

        # Text embedding: raw weight arrays (linear1/linear2 naming)
        scale1 = 1.0 / text_dim**0.5
        self.text_embedding_linear1_weight = mx.random.uniform(
            low=-scale1, high=scale1, shape=(dim, text_dim)
        )
        self.text_embedding_linear1_bias = mx.zeros((dim,))
        scale2 = 1.0 / dim**0.5
        self.text_embedding_linear2_weight = mx.random.uniform(
            low=-scale2, high=scale2, shape=(dim, dim)
        )
        self.text_embedding_linear2_bias = mx.zeros((dim,))

        # Time embedding: raw weight arrays (linear1/linear2 naming)
        scale_t1 = 1.0 / freq_dim**0.5
        self.time_embedding_linear1_weight = mx.random.uniform(
            low=-scale_t1, high=scale_t1, shape=(dim, freq_dim)
        )
        self.time_embedding_linear1_bias = mx.zeros((dim,))
        scale_t2 = 1.0 / dim**0.5
        self.time_embedding_linear2_weight = mx.random.uniform(
            low=-scale_t2, high=scale_t2, shape=(dim, dim)
        )
        self.time_embedding_linear2_bias = mx.zeros((dim,))

        # Time projection: raw weight arrays
        self.time_projection_linear_weight = mx.random.uniform(
            low=-scale_t2, high=scale_t2, shape=(6 * dim, dim)
        )
        self.time_projection_linear_bias = mx.zeros((6 * dim,))

        # Image embedding MLP for I2V: LayerNorm -> Linear -> GELU -> Linear -> LayerNorm
        if model_type == "i2v":
            clip_dim = 1280
            self.img_emb_norm1 = nn.LayerNorm(clip_dim)
            self.img_emb_linear1 = nn.Linear(clip_dim, clip_dim)
            self.img_emb_linear2 = nn.Linear(clip_dim, dim)
            self.img_emb_norm2 = nn.LayerNorm(dim)

        # Transformer blocks via setattr
        for i in range(num_layers):
            block = WanAttentionBlock(
                dim,
                ffn_dim,
                num_heads,
                cross_attn_norm,
                eps,
                cross_attn_type=model_type,
            )
            setattr(self, f"block_{i}", block)

        # Output head
        self.head = Head(dim, out_dim, patch_size, eps)

        # Precompute RoPE frequencies (not saved in checkpoint)
        self._freqs = precompute_rope_freqs(
            max_frames=1024,
            max_height=1024,
            max_width=1024,
            head_dim=self.head_dim,
            theta=10000.0,
        )

    @property
    def freqs(self):
        return self._freqs

    def _embed_text(self, context: mx.array) -> mx.array:
        return _embed_text_fn(
            context,
            self.text_embedding_linear1_weight,
            self.text_embedding_linear1_bias,
            self.text_embedding_linear2_weight,
            self.text_embedding_linear2_bias,
        )

    def _embed_image(self, clip_fea: mx.array) -> mx.array:
        """Project CLIP features through img_emb MLP."""
        x = self.img_emb_norm1(clip_fea)
        x = self.img_emb_linear1(x)
        x = nn.gelu(x)
        x = self.img_emb_linear2(x)
        x = self.img_emb_norm2(x)
        return x

    def _embed_time(self, t: mx.array) -> mx.array:
        return _embed_time_fn(
            self.freq_dim,
            t,
            self.time_embedding_linear1_weight,
            self.time_embedding_linear1_bias,
            self.time_embedding_linear2_weight,
            self.time_embedding_linear2_bias,
        )

    def _project_time(self, e: mx.array) -> mx.array:
        return _project_time_fn(
            e,
            self.time_projection_linear_weight,
            self.time_projection_linear_bias,
        )

    def compute_time_embedding(self, t: mx.array):
        """Compute time embeddings for TeaCache. Returns (t_emb, e0).
        t_emb: [1, dim] (pre-projection, used by head)
        e0: [1, 6*dim] (projected, used for block modulation)"""
        t_emb = self._embed_time(t)
        e0 = self._project_time(t_emb)
        return t_emb, e0

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        context: mx.array,
        context_lens: Optional[int] = None,
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
            context_lens: Actual context length (before padding)
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
        x = x[None]
        x = mx.conv3d(x, self.patch_embedding_weight, stride=self.patch_size, padding=0)
        x = x + self.patch_embedding_bias[None, None, None, None, :]
        _, Fp, Hp, Wp, _ = x.shape
        grid_sizes = [[Fp, Hp, Wp]]
        x = x.reshape(1, Fp * Hp * Wp, self.dim)

        # Embed context: [L, C_text] -> [1, text_len, dim]
        if context.shape[0] < self.text_len:
            pad_len = self.text_len - context.shape[0]
            context = mx.concatenate(
                [context, mx.zeros((pad_len, context.shape[1]))], axis=0
            )
        context = self._embed_text(context[None])

        # Prepend projected CLIP features to context (I2V)
        if clip_fea is not None:
            clip_proj = self._embed_image(clip_fea)
            context = mx.concatenate([clip_proj, context], axis=1)

        if context_lens is not None:
            context_lens = mx.array([context_lens], dtype=mx.int32)

        # Time embedding
        if precomputed_time is not None:
            t_emb, e = precomputed_time[0], precomputed_time[1]
        else:
            t_emb = self._embed_time(t)
            e = self._project_time(t_emb)
        e = e.reshape(1, 6, self.dim)

        # Transformer blocks
        if block_residual is not None:
            x = x + block_residual
            new_residual = block_residual  # pass through (caller won't cache this)
        else:
            x_in = x
            for i in range(self.num_layers):
                block = getattr(self, f"block_{i}")
                x = block(x, e, grid_sizes, self.freqs, context, context_lens)
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

            # patch_embedding.weight/bias -> patch_embedding_weight/bias
            new_key = re.sub(r"patch_embedding\.(\w+)", r"patch_embedding_\1", new_key)

            # Transpose Conv3d weights for patch_embedding
            if (
                "patch_embedding" in new_key
                and "weight" in new_key
                and len(value.shape) == 5
            ):
                value = mx.transpose(value, (0, 2, 3, 4, 1))

            # blocks.N -> block_N
            new_key = re.sub(r"blocks\.(\d+)\.", r"block_\1.", new_key)

            # ffn.0 -> ffn_linear1, ffn.2 -> ffn_linear2
            new_key = re.sub(r"ffn\.0\.(\w+)", r"ffn_linear1.\1", new_key)
            new_key = re.sub(r"ffn\.2\.(\w+)", r"ffn_linear2.\1", new_key)

            # text_embedding.0/2 -> text_embedding_linear1/2_weight/bias
            new_key = re.sub(
                r"text_embedding\.0\.(\w+)", r"text_embedding_linear1_\1", new_key
            )
            new_key = re.sub(
                r"text_embedding\.2\.(\w+)", r"text_embedding_linear2_\1", new_key
            )

            # time_embedding.0/2 -> time_embedding_linear1/2_weight/bias
            new_key = re.sub(
                r"time_embedding\.0\.(\w+)", r"time_embedding_linear1_\1", new_key
            )
            new_key = re.sub(
                r"time_embedding\.2\.(\w+)", r"time_embedding_linear2_\1", new_key
            )

            # time_projection.1 -> time_projection_linear_weight/bias
            new_key = re.sub(
                r"time_projection\.1\.(\w+)", r"time_projection_linear_\1", new_key
            )

            # head.head.weight -> head.head_weight
            new_key = re.sub(r"head\.head\.(\w+)", r"head.head_\1", new_key)

            # img_emb.proj.N -> img_emb_* (I2V MLPProj)
            new_key = re.sub(r"img_emb\.proj\.0\.(\w+)", r"img_emb_norm1.\1", new_key)
            new_key = re.sub(r"img_emb\.proj\.1\.(\w+)", r"img_emb_linear1.\1", new_key)
            new_key = re.sub(r"img_emb\.proj\.3\.(\w+)", r"img_emb_linear2.\1", new_key)
            new_key = re.sub(r"img_emb\.proj\.4\.(\w+)", r"img_emb_norm2.\1", new_key)

            remapped[new_key] = value

        # Merge separate Q/K/V into QKV for self-attention,
        # and K/V into KV for cross-attention
        remapped = WanModel._merge_qkv_weights(remapped)
        return remapped

    @staticmethod
    def _merge_qkv_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Merge separate q/k/v weights into qkv (self-attn) and kv (cross-attn)."""
        merged = {}
        consumed = set()

        for key in weights:
            # Self-attention: merge q, k, v -> qkv
            m = re.match(r"(block_\d+\.self_attn)\.(q)\.(weight|bias)$", key)
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
            m = re.match(r"(block_\d+\.cross_attn)\.(k)\.(weight|bias)$", key)
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


def create_wan_model(model_size: str = "1.3B", **kwargs) -> WanModel:
    configs = {
        "1.3B": {
            "dim": 1536,
            "ffn_dim": 8960,
            "freq_dim": 256,
            "num_heads": 12,
            "num_layers": 30,
        },
        "14B": {
            "dim": 5120,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "num_heads": 40,
            "num_layers": 40,
        },
    }
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}")
    config = configs[model_size]
    config.update(kwargs)
    return WanModel(**config)
