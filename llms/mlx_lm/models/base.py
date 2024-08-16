import inspect
from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn


class KVCache:

    def __init__(self, head_dim, n_kv_heads):
        self.n_kv_heads = n_kv_heads
        if isinstance(head_dim, int):
            self.k_head_dim = self.v_head_dim = head_dim
        elif isinstance(head_dim, tuple) and len(head_dim) == 2:
            self.k_head_dim, self.v_head_dim = head_dim
        else:
            raise ValueError("head_dim must be an int or a tuple of two ints")
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 256

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B = keys.shape[0]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, self.n_kv_heads, n_steps * self.step, self.k_head_dim)
            v_shape = (B, self.n_kv_heads, n_steps * self.step, self.v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def state(self):
        return self.keys, self.values


class RotatingKVCache:

    def __init__(self, head_dim, n_kv_heads, max_size, keep=0, step=256):
        self.n_kv_heads = n_kv_heads
        if isinstance(head_dim, int):
            self.k_head_dim = self.v_head_dim = head_dim
        elif isinstance(head_dim, tuple) and len(head_dim) == 2:
            self.k_head_dim, self.v_head_dim = head_dim
        else:
            raise ValueError("head_dim must be an int or a tuple of two ints")
        self.keep = keep
        self.keys = None
        self.values = None
        self.offset = 0
        self.max_size = max_size
        self.step = step
        if max_size % step != 0:
            raise ValueError(
                f"max_size must be a multiple of step but got {max_size}"
                f" and {step}, respectively."
            )
        self._idx = 0

    def update_and_fetch(self, keys, values):
        prev = self.offset
        B, _, S = keys.shape[:3]

        # Grow the cache if possible
        if self.keys is None or (
            (prev + S) > self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            n_steps = (self.step + S - 1) // self.step
            new_size = min(n_steps * self.step, self.max_size - prev)
            k_shape = (B, self.n_kv_heads, new_size, self.k_head_dim)
            v_shape = (B, self.n_kv_heads, new_size, self.v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += S

        # Rotate if needed
        if self._idx == self.max_size:
            self._idx = self.keep

        # Handle updates bigger than the available cache size
        kept = min(self.keep, self._idx)
        if S > self.max_size - kept:
            if kept < self.keep:
                n = self.keep - kept
                self.keys[..., self._idx : self.keep, :] = keys[..., :n, :]
                self.values[..., self._idx : self.keep, :] = values[..., :n, :]
                self._idx = self.keep
            start = S - self.max_size + self.keep
            keys = keys[..., start:, :]
            values = values[..., start:, :]
            self.offset += start
            S = keys.shape[2]

        # Overwrite the end of the buffer with the start of the keys/values
        if self._idx + S > self.max_size:
            end = self.max_size - self._idx
            self.keys[..., self._idx :, :] = keys[..., :end, :]
            self.values[..., self._idx :, :] = values[..., :end, :]
            keys = keys[..., end:, :]
            values = values[..., end:, :]
            self._idx = self.keep
            S = keys.shape[2]

        # Overwrite the beginning of the buffer with the end of keys/values
        end = self._idx + S
        self.keys[..., self._idx : end, :] = keys
        self.values[..., self._idx : end, :] = values
        self._idx += S

        # If the buffer is still not full, slice off the end
        if self.offset < self.max_size:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        return self.keys, self.values

    def state(self):
        return self.keys, self.values


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


def create_additive_causal_mask(N: int, offset: int = 0):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9


def create_attention_mask(h: mx.array, cache: Optional[List[KVCache]] = None):
    T = h.shape[1]
    if T > 1:
        # Input consists of multiple tokens, create a causal mask so that prior
        # tokens do not give attention to later tokens. If a cache is in place
        # (because e.g. prompt reuse), offset the mask accordingly.
        offset = cache[0].offset if cache is not None and cache[0] is not None else 0
        mask = create_additive_causal_mask(T, offset)
        mask = mask.astype(h.dtype)
    else:
        mask = None
    return mask
