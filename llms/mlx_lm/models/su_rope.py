# Copyright © 2023-2024 Apple Inc.

import math
from typing import List, Union

import mlx.core as mx
import mlx.nn as nn


class SuScaledRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dims: int,
        base: float = 10000.0,
        max_position_embeddings: int = 131072,
        original_max_position_embeddings: int = 4096,
        short_factor: Union[List[float], float] = 1.0,
        long_factor: Union[List[float], float] = 1.0,
        short_mscale: float = None,
        long_mscale: float = None,
    ):
        """
        Phi3Su Scaled Rotary Embedding layer for Phi-3 models.

        Args:
            dims (int): The feature dimensions to be rotated.
            base (int, optional): Base for the exponential scaling.
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
            short_mscale (float, optional): Scale the input prior to embedding.
            long_mscale (float, optional): Scale the input prior to embedding.
        """
        super().__init__()
        freqs = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        self._freqs = mx.array(long_factor, dtype=mx.float32) * freqs
        self.original_max_position_embeddings = original_max_position_embeddings
        self.scale = long_mscale or math.sqrt(
            1
            + math.log(max_position_embeddings / original_max_position_embeddings)
            / math.log(original_max_position_embeddings)
        )

    def __call__(self, x, offset: int = 0):
        return mx.fast.rope(
            self.scale * x,
            x.shape[-1],
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )
