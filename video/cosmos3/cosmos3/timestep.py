"""Timestep embedding for Cosmos 3 diffusion generation.

Sinusoidal timestep embedding → MLP projection, applied to noisy tokens
via scatter-add (only noisy frames get timestep conditioning).
"""

import math

import mlx.core as mx
import mlx.nn as nn


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding with MLP projection.

    Maps scalar timesteps to hidden_size embeddings.
    """

    def __init__(self, hidden_size: int = 4096, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.linear_1 = nn.Linear(freq_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def _sinusoidal_embedding(self, timesteps: mx.array) -> mx.array:
        """Compute sinusoidal embeddings for timesteps.

        Args:
            timesteps: [batch] scalar timesteps

        Returns:
            [batch, freq_dim] sinusoidal embeddings
        """
        half_dim = self.freq_dim // 2
        freqs = mx.exp(
            -math.log(10000.0) * mx.arange(half_dim, dtype=mx.float32) / half_dim
        )
        # [batch, 1] * [1, half_dim] -> [batch, half_dim]
        args = mx.expand_dims(timesteps.astype(mx.float32), -1) * mx.expand_dims(freqs, 0)
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        return embedding

    def __call__(self, timesteps: mx.array) -> mx.array:
        """Compute timestep embeddings.

        Args:
            timesteps: [batch] scalar timesteps

        Returns:
            [batch, hidden_size] timestep embeddings
        """
        emb = self._sinusoidal_embedding(timesteps)
        emb = nn.silu(self.linear_1(emb))
        emb = self.linear_2(emb)
        return emb


def apply_timestep_to_noisy_tokens(
    hidden_states: mx.array,
    timestep_emb: mx.array,
    noisy_mask: mx.array,
) -> mx.array:
    """Apply timestep embeddings only to noisy (generation) tokens.

    Instead of scatter_add, we use broadcasting with a mask.

    Args:
        hidden_states: [batch, seq_len, hidden_size]
        timestep_emb: [batch, hidden_size]
        noisy_mask: [batch, seq_len] boolean mask (True = noisy token)

    Returns:
        hidden_states with timestep embedding added to noisy positions
    """
    # Expand timestep_emb: [batch, 1, hidden_size]
    emb = mx.expand_dims(timestep_emb, 1)
    # Expand mask: [batch, seq_len, 1]
    mask = mx.expand_dims(noisy_mask.astype(hidden_states.dtype), -1)
    # Add only to masked positions
    return hidden_states + emb * mask
