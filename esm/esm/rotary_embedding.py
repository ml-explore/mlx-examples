from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


def rotate_half(x: mx.array) -> mx.array:
    """
    Rotate last dimension by splitting into two halves and swapping.

    Args:
        x: Tensor with even-sized last dimension.

    Returns:
        mx.array: Tensor of same shape as `x` with halves rotated.
    """
    # Split into two equal halves along the last dimension
    x1, x2 = mx.split(x, 2, axis=-1)
    # Swap halves and negate the second half
    return mx.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """
    Apply rotary position embeddings to a tensor.

    Args:
        x: Input tensor of shape (..., seq_len, dim).
        cos: Cosine embedding table of shape (1, seq_len, dim).
        sin: Sine embedding table of shape (1, seq_len, dim).

    Returns:
        mx.array: Tensor with rotary position embeddings applied.
    """
    # Trim cos/sin to match the sequence length of x
    cos = cos[:, : x.shape[-2], :]
    sin = sin[:, : x.shape[-2], :]

    # Elementwise rotation: (x * cos) + (rotate_half(x) * sin)
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """
    Rotary position embedding (RoPE) module.

    Args:
        dim (int): Head dimension size (must be even).
    """

    def __init__(self, dim: int, *_, **__):
        super().__init__()
        # Precompute inverse frequency for each pair of dimensions
        self.inv_freq = 1.0 / (10000 ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))

        # Cache for cosine/sine tables to avoid recomputation
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(
        self, x: mx.array, seq_dimension: int = 1
    ) -> Tuple[mx.array, mx.array]:
        """
        Compute and cache cos/sin tables for the given sequence length.

        Args:
            x: Reference tensor for sequence length.
            seq_dimension: Axis containing the sequence length.

        Returns:
            Tuple of:
                cos: Cosine table of shape (1, seq_len, dim).
                sin: Sine table of shape (1, seq_len, dim).
        """
        seq_len = x.shape[seq_dimension]
        # Only update cache if sequence length has changed
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            # Time steps: shape (seq_len,)
            t = mx.arange(seq_len).astype(self.inv_freq.dtype)
            # Outer product between time and inverse frequency
            freqs = mx.einsum("i,j->ij", t, self.inv_freq)
            # Duplicate frequencies for cos/sin dimensions
            emb = mx.concatenate((freqs, freqs), axis=-1)

            self._cos_cached = mx.cos(emb)[None, :, :]
            self._sin_cached = mx.sin(emb)[None, :, :]

        return self._cos_cached, self._sin_cached

    def __call__(self, q: mx.array, k: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Apply rotary position embeddings to queries and keys.

        Args:
            q: Query tensor of shape (..., seq_len, dim).
            k: Key tensor of shape (..., seq_len, dim).

        Returns:
            Tuple of:
                q_rot: Query tensor with RoPE applied.
                k_rot: Key tensor with RoPE applied.
        """
        # Get (and cache) cos/sin tables based on key sequence length
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            k, seq_dimension=-2
        )

        # Apply rotary embeddings to both q and k
        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )
