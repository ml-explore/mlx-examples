import math
from typing import List, Union

import mlx.core as mx


class SuScaledRotaryEmbedding:
    def __init__(
        self,
        dims: int,
        traditional: bool = False,
        base: float = 10000.0,
        scale: float = 1.0,
        max_position_embeddings: int = 131072,
        original_max_position_embeddings: int = 4096,
        short_factor: Union[List[float], float] = 1.0,
        long_factor: Union[List[float], float] = 1.0,
    ):
        """
        Phi3Su Scaled Rotary Embedding layer for Phi-3 models.

        Args:
            dims (int): The feature dimensions to be rotated.
            traditional (bool, optional): Unused. Default: ``False``.
            base (int, optional): Base for the exponential scaling.
            scale (float, optional): The scale used to scale the positions.
              Default: ``1.0``.
            max_position_embeddings (int, optional): The maximum sequence
              length that this model was trained with. This is used to determine
              the size of the original RoPE embeddings when using long scaling.
              Default: ``131072``.
            original_max_position_embeddings (int, optional): The maximum
              sequence length that this model was trained with. This is used to
              determine the size of the original RoPE embeddings when using long
              scaling. Default: ``4096``.
            short_factor (float or list[float], optional): List of scaling
              factors for sequences of length lesser than
              ``original_max_position_embeddings``. Default: ``1.0``.
            long_factor (float or list[float], optional): List of scaling
              factors for sequences of length greater than
              ``original_max_position_embeddings``.  Default: ``1.0``.
        """
        self.inv_freq_short = 1.0 / (
            mx.array(short_factor, dtype=mx.float32)
            * base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        )
        self.inv_freq_long = 1.0 / (
            scale
            * mx.array(long_factor, dtype=mx.float32)
            * base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        )
        self.original_max_position_embeddings = original_max_position_embeddings
        self.scaling_factor = math.sqrt(
            1
            + math.log(max_position_embeddings / original_max_position_embeddings)
            / math.log(original_max_position_embeddings)
        )

    def _get_cos_sin(self, offset, L):
        position_ids = mx.arange(offset, offset + L, dtype=mx.float32)
        inv_freq = (
            self.inv_freq_long
            if (offset + L) > self.original_max_position_embeddings
            else self.inv_freq_short
        )
        freqs = position_ids[:, None] * inv_freq[None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb) * self.scaling_factor
        sin = mx.sin(emb) * self.scaling_factor
        return cos, sin

    def __call__(self, x, offset: int = 0):
        def _rotate_half(_x):
            midpoint = _x.shape[-1] // 2
            x1, x2 = _x[..., :midpoint], _x[..., midpoint:]
            return mx.concatenate([-x2, x1], axis=-1)

        cos, sin = self._get_cos_sin(offset, x.shape[2])
        return (x * cos) + (_rotate_half(x) * sin)
