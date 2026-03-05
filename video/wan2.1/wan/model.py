# Copyright © 2026 Apple Inc.

"""
Wan2.1 non-causal DiT (Diffusion Transformer) for text-to-video generation.

Supports 1.3B and 14B model sizes. Uses bidirectional attention (no KV cache)
with setattr-based block registration for weight remapping compatibility.
"""

import math
import re
from functools import partial
from typing import Dict, List, Optional, Tuple

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
        qk_norm: bool = True,
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

        # Transformer blocks via setattr
        for i in range(num_layers):
            block = WanAttentionBlock(
                dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps
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

    def __call__(
        self,
        x: List[mx.array],
        t: mx.array,
        context: List[mx.array],
        context_lens: Optional[List[int]] = None,
    ) -> List[mx.array]:
        """
        Forward pass.

        Args:
            x: List of input latents, each [C_in, F, H, W]
            t: Timesteps [B]
            context: List of text embeddings, each [L, C_text]
            context_lens: Actual context lengths (before padding)

        Returns:
            List of output latents, each [C_out, F, H, W]
        """
        B = len(x)

        # Patchify and embed
        x_embedded = []
        grid_sizes = []
        seq_lens = []
        for x_i in x:
            x_i = x_i.transpose(1, 2, 3, 0)[None, :, :, :, :]  # [1, F, H, W, C]
            x_i = mx.conv3d(
                x_i, self.patch_embedding_weight, stride=self.patch_size, padding=0
            )
            x_i = x_i + self.patch_embedding_bias[None, None, None, None, :]
            _, Fp, Hp, Wp, _ = x_i.shape
            x_i = x_i.reshape(Fp * Hp * Wp, self.dim)
            x_embedded.append(x_i)
            grid_sizes.append([Fp, Hp, Wp])
            seq_lens.append(Fp * Hp * Wp)

        # Pad and stack into batch
        max_len = max(seq_lens)
        x_padded = []
        for x_i in x_embedded:
            if x_i.shape[0] < max_len:
                pad_len = max_len - x_i.shape[0]
                x_i = mx.concatenate([x_i, mx.zeros((pad_len, self.dim))], axis=0)
            x_padded.append(x_i)
        x = mx.stack(x_padded, axis=0)

        # Pad and embed context
        context_padded = []
        for c_i in context:
            if c_i.shape[0] < self.text_len:
                pad_len = self.text_len - c_i.shape[0]
                c_i = mx.concatenate([c_i, mx.zeros((pad_len, c_i.shape[1]))], axis=0)
            context_padded.append(c_i)
        context_padded = mx.stack(context_padded, axis=0)
        context = self._embed_text(context_padded)

        if context_lens is not None:
            context_lens = mx.array(context_lens, dtype=mx.int32)

        # Time embedding (per-sample, not per-patch)
        t_emb = self._embed_time(t).astype(mx.float32)  # [B, dim]
        e = self._project_time(t_emb).astype(mx.float32)
        e = e.reshape(B, 6, self.dim)  # [B, 6, dim]

        # Transformer blocks
        for i in range(self.num_layers):
            block = getattr(self, f"block_{i}")
            x = block(x, e, grid_sizes, self.freqs, context, context_lens)

        # Output head
        x = self.head(x, t_emb)

        # Unpatchify
        outputs = []
        for i, (seq_len_i, grid_size) in enumerate(zip(seq_lens, grid_sizes)):
            x_i = x[i, :seq_len_i, :]
            Fp, Hp, Wp = grid_size
            pt, ph, pw = self.patch_size
            x_i = rearrange(
                x_i,
                "(Fp Hp Wp) (pt ph pw c) -> c (Fp pt) (Hp ph) (Wp pw)",
                Fp=Fp,
                Hp=Hp,
                Wp=Wp,
                pt=pt,
                ph=ph,
                pw=pw,
            )
            outputs.append(x_i)

        return outputs

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Remap PyTorch checkpoint keys to MLX model format."""
        remapped = {}
        for key, value in weights.items():
            new_key = key

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

            remapped[new_key] = value
        return remapped


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
