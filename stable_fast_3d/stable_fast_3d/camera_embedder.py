"""
LinearCameraEmbedder + TriplaneLearnablePositionalEmbedding, the two
trivial, attention-free modules feeding into the DINOv2 tokenizer and
transformer backbone.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class LinearCameraEmbedder(nn.Module):
    """sf3d.models.camera.LinearCameraEmbedder — in=25 (flattened
    c2w_cond[16] + intrinsic_normed_cond[9]), out=768."""

    def __init__(self, in_channels: int = 25, out_channels: int = 768):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def __call__(self, c2w_cond: mx.array, intrinsic_normed_cond: mx.array) -> mx.array:
        # cond tensors arrive as (B, Nv, ...); flatten trailing dims and
        # concat in the SAME order as sf3d.Config.camera_embedder.conditions
        # (["c2w_cond", "intrinsic_normed_cond"]) - order matters, it's just
        # a flat concat before the linear layer.
        b, nv = c2w_cond.shape[:2]
        c2w_flat = c2w_cond.reshape(b, nv, -1)
        intr_flat = intrinsic_normed_cond.reshape(b, nv, -1)
        cond = mx.concatenate([c2w_flat, intr_flat], axis=-1)
        return self.linear(cond)


class TriplaneLearnablePositionalEmbedding(nn.Module):
    """sf3d.models.tokenizers.triplane.TriplaneLearnablePositionalEmbedding
    — a trainable [3, Ct, Hp, Wp] parameter, repeated per batch and
    reshaped. No real computation, just correct reshape order."""

    def __init__(self, plane_size: int = 96, num_channels: int = 1024):
        super().__init__()
        self.plane_size = plane_size
        self.num_channels = num_channels
        self.embeddings = mx.zeros((3, num_channels, plane_size, plane_size))

    def __call__(self, batch_size: int) -> mx.array:
        # einops: repeat 'Np Ct Hp Wp -> B Np Ct Hp Wp' then
        # rearrange 'B Np Ct Hp Wp -> B Ct (Np Hp Wp)'
        emb = mx.broadcast_to(
            self.embeddings[None],
            (batch_size, 3, self.num_channels, self.plane_size, self.plane_size),
        )
        emb = mx.transpose(emb, (0, 2, 1, 3, 4))  # B Ct Np Hp Wp
        return emb.reshape(
            batch_size, self.num_channels, 3 * self.plane_size * self.plane_size
        )
