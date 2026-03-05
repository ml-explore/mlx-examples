# Copyright © 2026 Apple Inc.

"""
Rotary Position Embedding (RoPE) for 3D video transformers.

Implements 3-axis RoPE for temporal, height, and width dimensions.
Uses mx.fast.rope for optimized Metal kernel.
"""

from functools import partial
from typing import Tuple

import mlx.core as mx
from einops import rearrange


def get_rope_dimensions(head_dim: int) -> Tuple[int, int, int]:
    """
    Get the dimension split for 3D RoPE.

    - Frame: d - 4*(d//6)
    - Height: 2*(d//6)
    - Width: 2*(d//6)
    """
    d = head_dim
    frame_dim = d - 4 * (d // 6)
    height_dim = 2 * (d // 6)
    width_dim = 2 * (d // 6)
    return frame_dim, height_dim, width_dim


def precompute_rope_freqs(
    max_frames: int,
    max_height: int,
    max_width: int,
    head_dim: int,
    theta: float = 10000.0,
) -> dict:
    """
    Precompute RoPE frequencies for 3D positions.

    Each axis gets its own frequency computation with its own dimension.
    """
    frame_dim, height_dim, width_dim = get_rope_dimensions(head_dim)

    dim_frame = frame_dim // 2
    dim_height = height_dim // 2
    dim_width = width_dim // 2

    frame_inv_freq = 1.0 / (
        theta ** (mx.arange(0, frame_dim, 2, dtype=mx.float32) / frame_dim)
    )
    height_inv_freq = 1.0 / (
        theta ** (mx.arange(0, height_dim, 2, dtype=mx.float32) / height_dim)
    )
    width_inv_freq = 1.0 / (
        theta ** (mx.arange(0, width_dim, 2, dtype=mx.float32) / width_dim)
    )

    frame_positions = mx.arange(max_frames, dtype=mx.float32)
    height_positions = mx.arange(max_height, dtype=mx.float32)
    width_positions = mx.arange(max_width, dtype=mx.float32)

    frame_freqs = frame_positions[:, None] * frame_inv_freq[None, :]
    frame_cos, frame_sin = mx.cos(frame_freqs), mx.sin(frame_freqs)

    height_freqs = height_positions[:, None] * height_inv_freq[None, :]
    height_cos, height_sin = mx.cos(height_freqs), mx.sin(height_freqs)

    width_freqs = width_positions[:, None] * width_inv_freq[None, :]
    width_cos, width_sin = mx.cos(width_freqs), mx.sin(width_freqs)

    return {
        "frame": {
            "cos": frame_cos,
            "sin": frame_sin,
            "dim": dim_frame,
            "full_dim": frame_dim,
        },
        "height": {
            "cos": height_cos,
            "sin": height_sin,
            "dim": dim_height,
            "full_dim": height_dim,
        },
        "width": {
            "cos": width_cos,
            "sin": width_sin,
            "dim": dim_width,
            "full_dim": width_dim,
        },
        "theta": theta,
        "head_dim": head_dim,
    }


@partial(mx.compile)
def _rope_3d(x, f, h, w, frame_dim, height_dim, width_dim, theta):
    B = x.shape[0]

    x_frame = x[..., :frame_dim]
    x_height = x[..., frame_dim : frame_dim + height_dim]
    x_width = x[..., frame_dim + height_dim :]

    # Frame RoPE
    x_frame = rearrange(x_frame, "B (f hw) n d -> (B hw) n f d", f=f)
    x_frame = mx.fast.rope(
        x_frame, dims=frame_dim, traditional=True, base=theta, scale=1.0, offset=0
    )
    x_frame = rearrange(x_frame, "(B hw) n f d -> B (f hw) n d", B=B, f=f)

    # Height RoPE
    x_height = rearrange(x_height, "B (f h w) n d -> (B f w) n h d", f=f, h=h, w=w)
    x_height = mx.fast.rope(
        x_height, dims=height_dim, traditional=True, base=theta, scale=1.0, offset=0
    )
    x_height = rearrange(x_height, "(B f w) n h d -> B (f h w) n d", B=B, f=f, w=w)

    # Width RoPE
    x_width = rearrange(x_width, "B (f h w) n d -> (B f h) n w d", f=f, h=h, w=w)
    x_width = mx.fast.rope(
        x_width, dims=width_dim, traditional=True, base=theta, scale=1.0, offset=0
    )
    x_width = rearrange(x_width, "(B f h) n w d -> B (f h w) n d", B=B, f=f, h=h)

    return mx.concatenate([x_frame, x_height, x_width], axis=-1)


def rope_apply(
    x: mx.array,
    grid_sizes: list,
    freqs: dict,
) -> mx.array:
    """
    Apply 3D RoPE using mx.fast.rope with reshapes.

    Args:
        x: Tensor of shape [B, L, H, D]
        grid_sizes: List of [frames, height, width] per batch element
        freqs: Precomputed frequencies from precompute_rope_freqs()

    Returns:
        Rotated tensor with same shape as x
    """
    f, h, w = grid_sizes[0]

    theta = freqs["theta"]
    frame_dim = freqs["frame"]["full_dim"]
    height_dim = freqs["height"]["full_dim"]
    width_dim = freqs["width"]["full_dim"]

    return _rope_3d(x, f, h, w, frame_dim, height_dim, width_dim, theta)
