# Copyright © 2026 Apple Inc.

"""
Rotary Position Embedding (RoPE) for 3D video transformers.

Implements 3-axis RoPE for temporal, height, and width dimensions.
Uses mx.fast.rope for optimized Metal kernel.
"""

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


@mx.compile
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
    head_dim: int,
    theta: float = 10000.0,
) -> mx.array:
    """
    Apply 3D RoPE using mx.fast.rope with reshapes.

    Args:
        x: Tensor of shape [B, L, H, D]
        grid_sizes: List of [frames, height, width] per batch element
        head_dim: Dimension per attention head
        theta: RoPE base frequency

    Returns:
        Rotated tensor with same shape as x
    """
    f, h, w = grid_sizes[0]

    frame_dim, height_dim, width_dim = get_rope_dimensions(head_dim)

    return _rope_3d(x, f, h, w, frame_dim, height_dim, width_dim, theta)
