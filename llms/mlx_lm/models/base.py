import inspect
from dataclasses import dataclass

import mlx.core as mx


class KVCache:

    def __init__(self, length, head_dim, n_kv_heads):
        self.keys = mx.zeros((1, n_kv_heads, length, head_dim))
        self.values = mx.zeros((1, n_kv_heads, length, head_dim))
        self.offset = 0

    def update_and_fetch(self, keys, values):
        prev = self.offset
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
