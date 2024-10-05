# Copyright © 2023-2024 Apple Inc.

import inspect
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def create_causal_mask(N: int, offset: int = 0, window_size: Optional[int] = None):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds < rinds
    if window_size is not None:
        mask = mask | (linds > rinds + window_size)
    return mask * -1e9


def create_attention_mask(h: mx.array, cache: Optional[Any] = None):
    T = h.shape[1]
    if T > 1:
        window_size = None
        offset = 0
        if cache is not None and cache[0] is not None:
            c = cache[0]
            if hasattr(c, "max_size"):
                offset = min(c.max_size - 1, c.offset)
                window_size = c.max_size
            else:
                offset = c.offset
        mask = create_causal_mask(T, offset, window_size=window_size)
        mask = mask.astype(h.dtype)
    else:
        mask = None
    return mask
