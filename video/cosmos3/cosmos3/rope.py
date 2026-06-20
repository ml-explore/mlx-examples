"""3D Multi-dimensional Rotary Position Embeddings for Cosmos 3.

Cosmos 3 uses interleaved mRoPE with three axes (temporal, height, width)
following the Qwen3-VL design. All axes share a single set of inverse
frequencies. Position IDs from different axes are interleaved before
frequency computation:
  - temporal positions go to indices {0, 3, 6, ...}
  - height positions go to indices {1, 4, 7, ...}
  - width positions go to indices {2, 5, 8, ...}

The mrope_section [24, 20, 20] determines how many frequency dimensions
are assigned to each axis. Total = 64 half-dims = 128 head_dim / 2.
"""

import math
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


class Cosmos3RotaryEmbedding(nn.Module):
    """Compute 3D interleaved mRoPE cos/sin embeddings.

    Matches HuggingFace Cosmos3VLTextRotaryEmbedding:
    - Single shared inv_freq for all axes
    - Interleaved axis layout (T,H,W at strides of 3)
    """

    def __init__(
        self,
        head_dim: int = 128,
        mrope_section: list[int] | None = None,
        rope_theta: float = 5_000_000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.mrope_section = mrope_section or [24, 20, 20]
        self.rope_theta = rope_theta

        half_dim = head_dim // 2
        assert sum(self.mrope_section) == half_dim, (
            f"mrope_section {self.mrope_section} sums to {sum(self.mrope_section)}, "
            f"expected {half_dim}"
        )

        # Single shared inv_freq for all axes (matching HF reference)
        inv_freq = 1.0 / (rope_theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
        self._inv_freq = inv_freq  # [half_dim = 64]

    def __call__(
        self,
        position_ids: mx.array,
        seq_len: int,
    ) -> Tuple[mx.array, mx.array]:
        """Compute cos/sin embeddings from 3-axis position IDs.

        Args:
            position_ids: [3, batch, seq_len] — position IDs per axis
            seq_len: sequence length

        Returns:
            cos: [batch, seq_len, head_dim]
            sin: [batch, seq_len, head_dim]
        """
        half_dim = self.head_dim // 2  # 64

        # Compute freqs for each axis: pos[axis] @ inv_freq
        # Each axis gets [batch, seq_len, half_dim] frequencies
        # freqs shape: [3, batch, seq_len, half_dim]
        freqs_per_axis = []
        for axis_idx in range(3):
            pos = position_ids[axis_idx].astype(mx.float32)  # [batch, seq_len]
            # Outer product: [batch, seq_len, 1] * [1, 1, half_dim]
            f = mx.expand_dims(pos, -1) * mx.expand_dims(
                mx.expand_dims(self._inv_freq, 0), 0
            )  # [batch, seq_len, half_dim]
            freqs_per_axis.append(f)

        # Interleave axes into a single frequency tensor
        # HF reference: temporal at {0,3,6,...}, height at {1,4,7,...}, width at {2,5,8,...}
        #
        # Build a mapping: for each output dim, which axis provides its value
        # All three axes computed the same 64 frequencies (shared inv_freq),
        # so we just need to pick which axis's position-scaled result goes where.
        #
        # Start with temporal everywhere, then overwrite H and W at stride-3
        # This matches HF's apply_interleaved_mrope
        axis_assignment = [0] * half_dim  # default: temporal
        for dim_offset, axis_idx in enumerate([1, 2], start=1):
            section_len = self.mrope_section[axis_idx]
            total_interleaved = section_len * 3
            for freq_idx in range(dim_offset, min(total_interleaved, half_dim), 3):
                axis_assignment[freq_idx] = axis_idx

        # Gather from the appropriate axis for each frequency dimension
        # Stack all axes: [3, batch, seq_len, half_dim]
        all_freqs = mx.stack(freqs_per_axis)  # [3, B, N, 64]
        # Select per-dim: use axis_assignment to index into axis dimension
        axis_indices = mx.array(axis_assignment)  # [half_dim]
        # Gather: for each dim d, take all_freqs[axis_assignment[d], :, :, d]
        freqs_parts = []
        for d in range(half_dim):
            a = axis_assignment[d]
            freqs_parts.append(freqs_per_axis[a][..., d:d+1])
        freqs = mx.concatenate(freqs_parts, axis=-1)  # [batch, seq_len, half_dim]

        # Double-up: [cos(f), cos(f)] and [sin(f), sin(f)] for full head_dim
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)

        return cos, sin


def rotate_half(x: mx.array) -> mx.array:
    """Rotate halves: [x1, x2] -> [-x2, x1] where x1, x2 are each half_dim."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: [batch, seq_len, num_heads, head_dim]
        k: [batch, seq_len, num_kv_heads, head_dim]
        cos: [batch, seq_len, half_dim]
        sin: [batch, seq_len, half_dim]

    Returns:
        q_rotated, k_rotated with same shapes as inputs
    """
    # Duplicate cos/sin to match full head_dim: [cos, cos] for the two halves
    cos_full = mx.concatenate([cos, cos], axis=-1)  # [batch, seq_len, head_dim]
    sin_full = mx.concatenate([sin, sin], axis=-1)  # [batch, seq_len, head_dim]

    # Expand for broadcasting over heads: [batch, seq_len, 1, head_dim]
    cos_full = mx.expand_dims(cos_full, 2)
    sin_full = mx.expand_dims(sin_full, 2)

    q_rot = q * cos_full + rotate_half(q) * sin_full
    k_rot = k * cos_full + rotate_half(k) * sin_full

    return q_rot, k_rot
