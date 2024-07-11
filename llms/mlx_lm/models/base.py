import inspect
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


class DynamicNTKScalingRoPE(nn.Module):
    """Implements the rotary positional encoding with Dynamic NTK scaling."""

    def __init__(
        self,
        dims: int,
        max_position_embeddings: int = 2048,
        traditional: bool = False,
        base: float = 10000,
        scale: float = 1.0,
    ):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.original_base = base
        self.dims = dims
        self.traditional = traditional
        self.scale = scale

    def extra_repr(self):
        return f"{self.dims}, traditional={self.traditional}, max_position_embeddings={self.max_position_embeddings}, scaling_factor={self.scaling_factor}"

    def __call__(self, x, offset: int = 0):
        seq_len = x.shape[1] + offset
        if seq_len > self.max_position_embeddings:
            self.base = self.original_base * (
                (self.scale * seq_len / self.max_position_embeddings) - (self.scale - 1)
            ) ** (self.dims / (self.dims - 2))
        else:
            self.base = self.original_base

        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=self.base,
            scale=self.scale,
            offset=offset,
        )


def create_additive_causal_mask(N: int, offset: int = 0):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9


class KVCache:

    def __init__(self, head_dim, n_kv_heads):
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 256

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            shape = (1, self.n_kv_heads, n_steps * self.step, self.head_dim)
            new_k = mx.zeros(shape, keys.dtype)
            new_v = mx.zeros(shape, values.dtype)
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
