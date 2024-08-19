# Copyright © 2023-2024 Apple Inc.

import math
from typing import List, Union

import mlx.core as mx
import mlx.nn as nn


class SuScaledRotaryEmbedding(nn.Module):
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
        super().__init__()
        self._short_freqs = mx.array(short_factor, dtype=mx.float32) * base ** (
            mx.arange(0, dims, 2, dtype=mx.float32) / dims
        )
        self._long_freqs = (
            scale
            * mx.array(long_factor, dtype=mx.float32)
            * base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        )
        self.original_max_position_embeddings = original_max_position_embeddings
        self.scale = math.sqrt(
            1
            + math.log(max_position_embeddings / original_max_position_embeddings)
            / math.log(original_max_position_embeddings)
        )

    def __call__(self, x, offset: int = 0):
        freqs = (
            self._long_freqs
            if (offset + x.shape[2]) > self.original_max_position_embeddings
            else self._short_freqs
        )
        return mx.fast.rope(
            x,
            x.shape[-1],
            traditional=False,
            base=1.0,
            scale=self.scale,
            offset=offset,
            freqs=freqs,
        )
