# Copyright © 2026 Apple Inc.

"""
Wan2.1 VAE encoder and decoder.

Encodes video frames to latents and decodes latents to video frames
using chunked processing with causal temporal caching.
"""

import re
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .vae_layers import (
    AttentionBlock,
    CausalConv3d,
    Resample,
    ResidualBlock,
    create_cache_entry,
)


class Decoder3d(nn.Module):
    """
    VAE Decoder for video generation.

    Input: [B, T, H/8, W/8, z_dim] (channels-last)
    Output: [B, T*4, H, W, 3]
    """

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: Optional[List[int]] = None,
        num_res_blocks: int = 2,
        attn_scales: Optional[List[float]] = None,
        temporal_upsample: Optional[List[bool]] = None,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temporal_upsample is None:
            temporal_upsample = [True, True, False]

        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.temporal_upsample = temporal_upsample

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        self.middle_res1 = ResidualBlock(dims[0], dims[0])
        self.middle_attn = AttentionBlock(dims[0])
        self.middle_res2 = ResidualBlock(dims[0], dims[0])

        # Build upsample stages as nested lists
        self.upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            stage = []
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for j in range(num_res_blocks + 1):
                stage.append(ResidualBlock(in_dim, out_dim))
                if scale in attn_scales:
                    stage.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temporal_upsample[i] else "upsample2d"
                stage.append(Resample(out_dim, mode=mode))
                scale *= 2.0
            self.upsamples.append(stage)

        self.head_norm = nn.RMSNorm(dims[-1], eps=1e-12)
        self.head_conv = CausalConv3d(dims[-1], 3, 3, padding=1)

        # Count temporal cache slots from architecture
        n = 1 + 2 + 2  # conv1, middle_res1, middle_res2
        for stage in self.upsamples:
            for layer in stage:
                if isinstance(layer, ResidualBlock):
                    n += 2
                elif isinstance(layer, Resample) and hasattr(layer, "time_conv"):
                    n += 1
        n += 1  # head_conv
        self.num_cache_slots = n

    def __call__(self, x, feat_cache):
        cache_idx = 0
        new_cache = []

        cache_input = x
        x = self.conv1(x, feat_cache[cache_idx])
        new_cache.append(create_cache_entry(cache_input, feat_cache[cache_idx]))
        cache_idx += 1

        x, c1, c2 = self.middle_res1(
            x, feat_cache[cache_idx], feat_cache[cache_idx + 1]
        )
        new_cache.append(c1)
        new_cache.append(c2)
        cache_idx += 2

        x = self.middle_attn(x)

        x, c1, c2 = self.middle_res2(
            x, feat_cache[cache_idx], feat_cache[cache_idx + 1]
        )
        new_cache.append(c1)
        new_cache.append(c2)
        cache_idx += 2

        for stage in self.upsamples:
            for layer in stage:
                if isinstance(layer, ResidualBlock):
                    x, c1, c2 = layer(
                        x, feat_cache[cache_idx], feat_cache[cache_idx + 1]
                    )
                    new_cache.append(c1)
                    new_cache.append(c2)
                    cache_idx += 2
                elif isinstance(layer, AttentionBlock):
                    x = layer(x)
                elif isinstance(layer, Resample):
                    x, c = layer(x, feat_cache[cache_idx])
                    if c is not None:
                        new_cache.append(c)
                        cache_idx += 1

        x = self.head_norm(x)
        x = nn.silu(x)
        cache_input = x
        x = self.head_conv(x, feat_cache[cache_idx])
        new_cache.append(create_cache_entry(cache_input, feat_cache[cache_idx]))
        cache_idx += 1

        return x, new_cache


class Encoder3d(nn.Module):
    """
    VAE Encoder for video generation.

    Input: [B, T, H, W, 3] (channels-last)
    Output: [B, T', H/8, W/8, z_dim*2]
    """

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: Optional[List[int]] = None,
        num_res_blocks: int = 2,
        attn_scales: Optional[List[float]] = None,
        temporal_downsample: Optional[List[bool]] = None,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temporal_downsample is None:
            temporal_downsample = [False, True, True]

        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.temporal_downsample = temporal_downsample

        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # Build downsample stages as nested lists
        self.downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            stage = []
            for j in range(num_res_blocks):
                stage.append(ResidualBlock(in_dim, out_dim))
                if scale in attn_scales:
                    stage.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temporal_downsample[i] else "downsample2d"
                stage.append(Resample(out_dim, mode=mode))
                scale /= 2.0
            self.downsamples.append(stage)

        self.middle_res1 = ResidualBlock(dims[-1], dims[-1])
        self.middle_attn = AttentionBlock(dims[-1])
        self.middle_res2 = ResidualBlock(dims[-1], dims[-1])

        self.head_norm = nn.RMSNorm(dims[-1], eps=1e-12)
        self.head_conv = CausalConv3d(dims[-1], z_dim * 2, 3, padding=1)

        # Count temporal cache slots from architecture
        n = 1  # conv1
        for stage in self.downsamples:
            for layer in stage:
                if isinstance(layer, ResidualBlock):
                    n += 2
                elif isinstance(layer, Resample) and hasattr(layer, "time_conv"):
                    n += 1
        n += 2 + 2 + 1  # middle_res1, middle_res2, head_conv
        self.num_cache_slots = n

    def __call__(self, x, feat_cache):
        cache_idx = 0
        new_cache = []

        cache_input = x
        x = self.conv1(x, feat_cache[cache_idx])
        new_cache.append(create_cache_entry(cache_input, feat_cache[cache_idx]))
        cache_idx += 1

        for stage in self.downsamples:
            for layer in stage:
                if isinstance(layer, ResidualBlock):
                    x, c1, c2 = layer(
                        x, feat_cache[cache_idx], feat_cache[cache_idx + 1]
                    )
                    new_cache.append(c1)
                    new_cache.append(c2)
                    cache_idx += 2
                elif isinstance(layer, AttentionBlock):
                    x = layer(x)
                elif isinstance(layer, Resample):
                    x, c = layer(x, feat_cache[cache_idx])
                    if c is not None:
                        new_cache.append(c)
                        cache_idx += 1

        x, c1, c2 = self.middle_res1(
            x, feat_cache[cache_idx], feat_cache[cache_idx + 1]
        )
        new_cache.append(c1)
        new_cache.append(c2)
        cache_idx += 2

        x = self.middle_attn(x)

        x, c1, c2 = self.middle_res2(
            x, feat_cache[cache_idx], feat_cache[cache_idx + 1]
        )
        new_cache.append(c1)
        new_cache.append(c2)
        cache_idx += 2

        x = self.head_norm(x)
        x = nn.silu(x)
        cache_input = x
        x = self.head_conv(x, feat_cache[cache_idx])
        new_cache.append(create_cache_entry(cache_input, feat_cache[cache_idx]))
        cache_idx += 1

        return x, new_cache


class WanVAE(nn.Module):
    """
    High-level VAE wrapper for Wan2.1.

    Encode: [F, H, W, C] video -> [F', H/8, W/8, z_dim] latent
    Decode: [F, H, W, C] latent -> [F*4, H*8, W*8, 3] video clamped to [-1, 1]
    """

    def __init__(self):
        super().__init__()
        self.encoder = Encoder3d()
        self.conv1 = CausalConv3d(32, 32, 1)
        self.decoder = Decoder3d()
        self.conv2 = CausalConv3d(16, 16, 1)

        self.mean = mx.array(
            [
                -0.7571,
                -0.7089,
                -0.9113,
                0.1075,
                -0.1745,
                0.9653,
                -0.1517,
                1.5508,
                0.4134,
                -0.0715,
                0.5517,
                -0.3632,
                -0.1922,
                -0.9497,
                0.2503,
                -0.2921,
            ]
        )
        self.std = mx.array(
            [
                2.8184,
                1.4541,
                2.3275,
                2.6558,
                1.2196,
                1.7708,
                2.6052,
                2.0743,
                3.2687,
                2.1526,
                2.8652,
                1.5579,
                1.6382,
                1.1253,
                2.8251,
                1.9160,
            ]
        )
        self.z_dim = 16
        # Pre-compile for the frame-by-frame loop: avoids recompiling each frame.
        self._compiled_decode = mx.compile(self.decoder.__call__)
        self._compiled_encode = mx.compile(self.encoder.__call__)

    def decode(self, z: mx.array) -> mx.array:
        """
        Decode latent to video.

        Args:
            z: Latent tensor [F, H, W, C] (channels-last)

        Returns:
            Video tensor [F, H, W, C] clamped to [-1, 1] (channels-last)
        """
        # Add batch dim: [F, H, W, C] -> [1, F, H, W, C]
        z = z[None]

        # Unscale latents
        scale = 1.0 / self.std
        z = z / scale.reshape(1, 1, 1, 1, self.z_dim) + self.mean.reshape(
            1, 1, 1, 1, self.z_dim
        )

        # Pre-decoder conv
        x = self.conv2(z)

        # Decode one frame at a time. mx.eval per frame releases intermediates, keeping memory bounded.
        num_frames = x.shape[1]
        feat_cache = [None] * self.decoder.num_cache_slots
        outputs = []

        for i in range(num_frames):
            frame = x[:, i : i + 1, :, :, :]
            out_frame, feat_cache = self._compiled_decode(frame, feat_cache)
            mx.eval(out_frame)
            outputs.append(out_frame)

        out = mx.concatenate(outputs, axis=1)
        out = mx.clip(out, -1.0, 1.0)

        # Remove batch dim: [1, F, H, W, C] -> [F, H, W, C]
        return out[0]

    def encode(self, x: mx.array) -> mx.array:
        """
        Encode video to latent.

        Args:
            x: Video tensor [F, H, W, C] (channels-last)

        Returns:
            Latent tensor [F', H/8, W/8, C] (channels-last)
        """
        # Add batch dim: [F, H, W, C] -> [1, F, H, W, C]
        x = x[None]

        num_frames = x.shape[1]
        feat_cache = [None] * self.encoder.num_cache_slots
        outputs = []

        # First chunk is 1 frame (causal init), subsequent chunks are 4 frames (matching VAE temporal stride).
        i = 0
        chunk_idx = 0
        while i < num_frames:
            if chunk_idx == 0:
                chunk = x[:, i : i + 1, :, :, :]
                i += 1
            else:
                chunk = x[:, i : i + 4, :, :, :]
                i += 4

            out_chunk, feat_cache = self._compiled_encode(chunk, feat_cache)
            mx.eval(out_chunk)
            outputs.append(out_chunk)
            chunk_idx += 1

        out = mx.concatenate(outputs, axis=1)

        # Post-encoder conv and extract mu
        out = self.conv1(out)
        mu = out[:, :, :, :, : self.z_dim]

        # Scale: (mu - mean) * (1/std)
        scale = 1.0 / self.std
        mu = (mu - self.mean.reshape(1, 1, 1, 1, self.z_dim)) * scale.reshape(
            1, 1, 1, 1, self.z_dim
        )

        # Remove batch dim: [1, F', H', W', C] -> [F', H', W', C]
        return mu[0]

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Remap PyTorch VAE keys to MLX format."""
        remapped = {}
        for key, value in weights.items():
            new_key = key

            # Transpose convolution weights
            if "weight" in new_key:
                if len(value.shape) == 5:
                    value = mx.transpose(value, (0, 2, 3, 4, 1))
                elif len(value.shape) == 4:
                    value = mx.transpose(value, (0, 2, 3, 1))

            new_key = new_key.replace(".gamma", ".weight")
            new_key = new_key.replace("decoder.middle.0.", "decoder.middle_res1.")
            new_key = new_key.replace("decoder.middle.1.", "decoder.middle_attn.")
            new_key = new_key.replace("decoder.middle.2.", "decoder.middle_res2.")
            new_key = new_key.replace("decoder.head.0.", "decoder.head_norm.")
            new_key = new_key.replace("decoder.head.2.", "decoder.head_conv.")
            new_key = new_key.replace("encoder.middle.0.", "encoder.middle_res1.")
            new_key = new_key.replace("encoder.middle.1.", "encoder.middle_attn.")
            new_key = new_key.replace("encoder.middle.2.", "encoder.middle_res2.")
            new_key = new_key.replace("encoder.head.0.", "encoder.head_norm.")
            new_key = new_key.replace("encoder.head.2.", "encoder.head_conv.")

            if "decoder.upsamples." in new_key:
                new_key = _map_vae_upsample_key(new_key)
            if "encoder.downsamples." in new_key:
                new_key = _map_vae_downsample_key(new_key)

            new_key = re.sub(r"\.residual\.0\.", ".norm1.", new_key)
            new_key = re.sub(r"\.residual\.2\.", ".conv1.", new_key)
            new_key = re.sub(r"\.residual\.3\.", ".norm2.", new_key)
            new_key = re.sub(r"\.residual\.6\.", ".conv2.", new_key)

            # Resample conv: .resample.1. -> .conv.
            new_key = re.sub(r"\.resample\.1\.", ".conv.", new_key)

            # Squeeze 1x1 conv weights to 2D for nn.Linear (to_qkv, proj)
            if ("to_qkv" in new_key or "proj" in new_key) and "weight" in new_key:
                if (
                    len(value.shape) == 4
                    and value.shape[1] == 1
                    and value.shape[2] == 1
                ):
                    value = value.reshape(value.shape[0], value.shape[3])

            # Squeeze norm weights to 1D (required — nn.RMSNorm expects 1D)
            if "norm" in new_key and "weight" in new_key:
                if len(value.shape) > 1:
                    value = mx.squeeze(value)

            remapped[new_key] = value
        return remapped


def _map_vae_upsample_key(key: str) -> str:
    match = re.match(r"decoder\.upsamples\.(\d+)\.(.*)", key)
    if not match:
        return key

    layer_idx = int(match.group(1))
    rest = match.group(2)

    # Decoder stages: (num_res_blocks+1) ResBlocks + 1 Resample each, except last (no Resample).
    # Assumes attn_scales=[] (Wan2.1 default — no AttentionBlocks in stages).
    num_res_blocks, num_stages = 2, 4
    stage_sizes = [num_res_blocks + 2] * (num_stages - 1) + [num_res_blocks + 1]
    stage = 0
    local_idx = layer_idx

    for s, size in enumerate(stage_sizes):
        if local_idx < size:
            stage = s
            break
        local_idx -= size

    return f"decoder.upsamples.{stage}.{local_idx}.{rest}"


def _map_vae_downsample_key(key: str) -> str:
    match = re.match(r"encoder\.downsamples\.(\d+)\.(.*)", key)
    if not match:
        return key

    layer_idx = int(match.group(1))
    rest = match.group(2)

    # Encoder stages: num_res_blocks ResBlocks + 1 Resample each, except last (no Resample).
    # Assumes attn_scales=[] (Wan2.1 default — no AttentionBlocks in stages).
    num_res_blocks, num_stages = 2, 4
    stage_sizes = [num_res_blocks + 1] * (num_stages - 1) + [num_res_blocks]
    stage = 0
    local_idx = layer_idx

    for s, size in enumerate(stage_sizes):
        if local_idx < size:
            stage = s
            break
        local_idx -= size

    return f"encoder.downsamples.{stage}.{local_idx}.{rest}"
