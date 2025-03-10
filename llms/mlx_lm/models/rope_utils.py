# Copyright © 2023-2024 Apple Inc.

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class Llama3RoPE(nn.Module):

    def __init__(
        self,
        dims: int,
        max_position_embeddings: int = 2048,
        traditional: bool = False,
        base: float = 10000,
        scaling_config: dict = None,
    ):
        super().__init__()
        self.dims = dims
        self.max_position_embeddings = max_position_embeddings
        self.traditional = traditional

        factor = scaling_config["factor"]
        low_freq_factor = scaling_config.get("low_freq_factor", 1.0)
        high_freq_factor = scaling_config.get("high_freq_factor", 4.0)
        old_context_len = scaling_config.get(
            "original_max_position_embeddings",
            8192,
        )

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        freqs = base ** (mx.arange(0, dims, 2) / dims)
        wavelens = 2 * mx.pi * freqs

        freqs = mx.where(wavelens > low_freq_wavelen, freqs * factor, freqs)
        is_medium_freq = (wavelens > high_freq_wavelen) & (wavelens < low_freq_wavelen)
        smooth_factors = (old_context_len / wavelens - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smooth_freqs = freqs / ((1 - smooth_factors) / factor + smooth_factors)
        self._freqs = mx.where(is_medium_freq, smooth_freqs, freqs)

    def extra_repr(self):
        return (
            f"{self.dims}, traditional={self.traditional}, "
            f"max_position_embeddings={self.max_position_embeddings}"
        )

    def __call__(self, x, offset: int = 0):
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class YarnRoPE(nn.Module):
    def __init__(
        self,
        dims,
        traditional=False,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        super().__init__()

        def yarn_find_correction_dim(num_rotations):
            return (
                dims
                * math.log(
                    original_max_position_embeddings / (num_rotations * 2 * math.pi)
                )
            ) / (2 * math.log(base))

        def yarn_find_correction_range():
            low = math.floor(yarn_find_correction_dim(beta_fast))
            high = math.ceil(yarn_find_correction_dim(beta_slow))
            return max(low, 0), min(high, dims - 1)

        def yarn_get_mscale(scale=1, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        def yarn_linear_ramp_mask(min_val, max_val, dim):
            if min_val == max_val:
                max_val += 0.001  # Prevent singularity

            linear_func = (mx.arange(dim, dtype=mx.float32) - min_val) / (
                max_val - min_val
            )
            return mx.clip(linear_func, 0, 1)

        self.mscale = yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(
            scaling_factor, mscale_all_dim
        )
        freq_extra = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        freq_inter = scaling_factor * base ** (
            mx.arange(0, dims, 2, dtype=mx.float32) / dims
        )
        low, high = yarn_find_correction_range()
        freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dims // 2)
        self._freqs = (freq_inter * freq_extra) / (
            freq_inter * freq_mask + freq_extra * (1 - freq_mask)
        )
        self.dims = dims
        self.traditional = traditional

    def __call__(self, x, offset=0):
        if self.mscale != 1.0:
            x[..., : self.dims] = self.mscale * x[..., : self.dims]
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


def initialize_rope(
    dims,
    base,
    traditional,
    scaling_config: Optional[dict] = None,
    max_position_embeddings: Optional[int] = None,
):
    if scaling_config is not None:
        rope_type = scaling_config.get("type") or scaling_config.get(
            "rope_type", "default"
        )
    else:
        rope_type = "default"

    if rope_type in ["default", "linear"]:
        scale = 1 / scaling_config["factor"] if rope_type == "linear" else 1.0
        return nn.RoPE(dims, traditional=traditional, base=base, scale=scale)

    elif rope_type == "llama3":
        return Llama3RoPE(
            dims=dims,
            max_position_embeddings=max_position_embeddings,
            traditional=traditional,
            base=base,
            scaling_config=scaling_config,
        )
    elif rope_type == "yarn":
        scaling_factor = scaling_config["factor"]
        rope_kwargs = {
            key: scaling_config[key]
            for key in [
                "original_max_position_embeddings",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            ]
            if key in scaling_config
        }
        return YarnRoPE(
            dims=dims,
            max_position_embeddings=max_position_embeddings,
            traditional=traditional,
            base=base,
            **rope_kwargs,
        )
    else:
        raise ValueError(f"Unsupported RoPE type {rope_type}")
