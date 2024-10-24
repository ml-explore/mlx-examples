# Copyright © 2023-2024 Apple Inc.

from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


def make_prompt_cache(model: nn.Module, max_kv_size: Optional[int] = None) -> List[Any]:
    """
    Construct the model's cache for use when cgeneration.

    This function will defer the cache construction to the model if it has a
    ``make_cache`` method, otherwise it will make a default KV cache.

    Args:
        model (nn.Module): The language model.
        max_kv_size (Optional[int]): If provided and the model does not have a
            ``make_cache`` method, a ``RotatingKVCache`` is used with a maximum
            size of ``max_kv_size``
    """
    if hasattr(model, "make_cache"):
        return model.make_cache()

    num_layers = len(model.layers)
    if max_kv_size is not None:
        return [
            RotatingKVCache(max_size=max_kv_size, keep=4) for _ in range(num_layers)
        ]
    else:
        return [KVCache() for _ in range(num_layers)]


def save_prompt_cache(file_name: str, cache: List[Any], metadata: Dict[str, str] = {}):
    """
    Save a pre-computed prompt cache to a file.

    Args:
        file_name (str): The ``.safetensors`` file name.
        cache (List[Any]): The model state.
        metadata (Dict[str, str]): Optional metadata to save along with model
            state.
    """
    cache_data = [c.state for c in cache]
    cache_info = [c.meta_state for c in cache]
    cache_data = dict(tree_flatten(cache_data))
    cache_classes = [type(c).__name__ for c in cache]
    cache_metadata = [cache_info, metadata, cache_classes]
    cache_metadata = dict(tree_flatten(cache_metadata))
    mx.save_safetensors(file_name, cache_data, cache_metadata)


def load_prompt_cache(file_name, return_metadata=False):
    """
    Load a prompt cache from a file.

    Args:
        file_name (str): The ``.safetensors`` file name.
        return_metadata (bool): Whether or not to return metadata.
            Default: ``False``.

    Returns:
        List[Any] or Tuple[List[Any], Dict[str, str]]: The prompt cache and
            the metadata if requested.
    """
    arrays, cache_metadata = mx.load(file_name, return_metadata=True)
    arrays = tree_unflatten(list(arrays.items()))
    cache_metadata = tree_unflatten(list(cache_metadata.items()))
    info, metadata, classes = cache_metadata
    cache = [globals()[c]() for c in classes]
    for c, state, meta_state in zip(cache, arrays, info):
        c.state = state
        c.meta_state = meta_state
    if return_metadata:
        return cache, metadata
    return cache


def can_trim_prompt_cache(cache: List[Any]) -> bool:
    """
    Check if model's cache can be trimmed.
    """
    return all(c.is_trimmable() for c in cache)


def trim_prompt_cache(cache: List[Any], num_tokens: int) -> List[Any]:
    """
    Trim the model's cache by the given number of tokens.

    This function will trim the cache if possible (in-place) and return the
    number of tokens that were trimmed.

    Args:
        cache (List[Any]): The model's cache.
        num_tokens (int): The number of tokens to trim.

    Returns:
        (int): The number of tokens that were trimmed.
    """
    if not can_trim_prompt_cache(cache) or len(cache) == 0:
        return 0
    return [c.trim(num_tokens) for c in cache][0]


class _BaseCache:
    @property
    def state(self):
        return []

    @state.setter
    def state(self, v):
        if v is not None and v:
            raise ValueError("This cache has no state but a state was set.")

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        if v is not None and v:
            raise ValueError("This cache has no meta_state but a meta_state was set.")

    def is_trimmable(self):
        return False


class KVCache(_BaseCache):
    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 256

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
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

    @property
    def state(self):
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        else:
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = self.keys.shape[2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n


class RotatingKVCache(_BaseCache):

    def __init__(self, max_size=None, keep=0, step=256):
        self.keep = keep
        self.keys = None
        self.values = None
        self.offset = 0
        self.max_size = max_size
        self.step = step
        self._idx = 0

    def _trim(self, trim_size, v, append=None):
        to_cat = []
        if trim_size > 0:
            to_cat = [v[..., : self.keep, :], v[..., trim_size + self.keep :, :]]
        else:
            to_cat = [v]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)

    def _temporal_order(self, v):
        """
        Rearrange the cache into temporal order, slicing off the end if unused.
        """
        if self._idx == v.shape[2]:
            return v
        elif self._idx < self.offset:
            return mx.concatenate(
                [
                    v[..., : self.keep, :],
                    v[..., self._idx :, :],
                    v[..., self.keep : self._idx, :],
                ],
                axis=2,
            )
        else:
            return v[..., : self._idx, :]

    def _update_concat(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            # Put the keys/values in temporal order to
            # preserve context
            self.keys = self._temporal_order(self.keys)
            self.values = self._temporal_order(self.values)

            # The largest size is self.max_size + S - 1 to ensure
            # every token gets at least self.max_size context
            trim_size = self._idx - self.max_size + 1
            self.keys = self._trim(trim_size, self.keys, keys)
            self.values = self._trim(trim_size, self.values, values)
        self.offset += keys.shape[2]
        self._idx = self.keys.shape[2]
        return self.keys, self.values

    def _update_in_place(self, keys, values):
        # May not have hit the max size yet, so potentially
        # keep growing the cache
        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self.offset
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        # Trim if needed
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size

        # Rotate
        if self._idx == self.max_size:
            self._idx = self.keep

        # Assign
        self.keys[..., self._idx : self._idx + S, :] = keys
        self.values[..., self._idx : self._idx + S, :] = values
        self.offset += S
        self._idx += S

        # If the buffer is not full, slice off the end
        if self.offset < self.max_size:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        return self.keys, self.values

    def update_and_fetch(self, keys, values):
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    @property
    def state(self):
        if self.offset < self.keys.shape[2]:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        else:
            return self.keys, self.values

    @state.setter
    def state(self, v):
        self.keys, self.values = v

    @property
    def meta_state(self):
        return tuple(
            map(str, (self.keep, self.max_size, self.step, self.offset, self._idx))
        )

    @meta_state.setter
    def meta_state(self, v):
        self.keep, self.max_size, self.step, self.offset, self._idx = map(
            int,
            v,
        )

    def is_trimmable(self):
        return self.offset < self.max_size

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        self._idx -= n
        return n


class MambaCache(_BaseCache):
    def __init__(self):
        self.cache = [None, None]

    def __setitem__(self, idx, value):
        self.cache[idx] = value

    def __getitem__(self, idx):
        return self.cache[idx]

    @property
    def state(self):
        return self.cache

    @state.setter
    def state(self, v):
        self.cache = v


class Mamba2Cache:
    batch_size: int
    intermediate_size: int
    state_size: int
    conv_kernel: int
    num_heads: int
    head_dim: int
    
    def __init__(
        self,
        batch_size: int,
        intermediate_size: int,
        state_size: int,
        conv_kernel: int,
        num_heads: int,
        head_dim: int
    ):
        self.batch_size = batch_size
        self.intermediate_size = intermediate_size
        self.state_size = state_size
        self.conv_kernel = conv_kernel
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Initialize conv state with proper dimensions
        self.conv_dim = self.intermediate_size + 2 * self.state_size
        self.conv_state = mx.zeros((batch_size, self.conv_dim, conv_kernel - 1))
        
        # Initialize SSM state
        self.ssm_state = mx.zeros((
            batch_size,
            num_heads,
            head_dim,
            state_size
        ))

    def update_conv_state(self, x: mx.array) -> mx.array:
        """
        Update convolution state for incremental inference.
        Args:
            x: Input tensor containing projected values (B, conv_in_dim)
        Returns:
            Combined state tensor of shape (batch_size, conv_dim, kernel_size)
        """
        # Handle input shape
        if x.ndim == 1:
            x = mx.expand_dims(x, 0)  # Add batch dimension if needed
        
        # Ensure batch size matches
        assert x.shape[0] == self.batch_size, f"Batch size mismatch: {x.shape[0]} vs {self.batch_size}"
        
        # Reshape x to match conv_dim
        # The input x contains intermediate_size + 2 * state_size dimensions
        x_reshaped = mx.reshape(x, (self.batch_size, -1))
        x_padded = mx.pad(
            x_reshaped,
            [(0, 0), (0, self.conv_dim - x_reshaped.shape[1])],
            mode='constant',
            constant_values=0
        )
        
        # Expand dims for concatenation
        x_expanded = mx.expand_dims(x_padded, -1)  # Shape: (batch_size, conv_dim, 1)
        
        # Roll the existing state left by 1
        rolled_state = mx.roll(self.conv_state, shift=-1, axis=-1)
        
        # Create update mask for the last position
        update_pos = self.conv_kernel - 2
        state_idx = mx.arange(self.conv_kernel - 1)
        update_mask = state_idx == update_pos
        
        # Broadcast mask to match state dimensions
        update_mask = mx.broadcast_to(
            mx.reshape(update_mask, (1, 1, -1)),
            rolled_state.shape
        )
        
        # Update state with padded input
        x_broadcast = mx.broadcast_to(x_expanded, (self.batch_size, self.conv_dim, 1))
        self.conv_state = mx.where(
            update_mask,
            x_broadcast,
            rolled_state
        )
        
        # Return concatenated state for convolution
        return mx.concatenate([self.conv_state, x_expanded], axis=-1)

    def update_ssm_state(self, dA: mx.array, dBx: mx.array) -> mx.array:
        """
        Update SSM state for incremental inference.
        Args:
            dA: State transition tensor of shape (batch_size, num_heads)
            dBx: Input projection tensor of shape (batch_size, num_heads, head_dim, state_size)
        Returns:
            Updated SSM state of shape (batch_size, num_heads, head_dim, state_size)
        """
        # Add necessary dimensions to dA for broadcasting
        # dA shape: (batch_size, num_heads) -> (batch_size, num_heads, 1, 1)
        dA = mx.expand_dims(mx.expand_dims(dA, -1), -1)
        
        # Ensure dBx has the correct shape
        assert dBx.shape[-1] == self.state_size, f"dBx state dimension mismatch: {dBx.shape[-1]} vs {self.state_size}"
        assert dBx.shape[-2] == self.head_dim, f"dBx head dimension mismatch: {dBx.shape[-2]} vs {self.head_dim}"
        
        # Update state: state = dA * state + dBx
        self.ssm_state = dA * self.ssm_state + dBx
        
        return self.ssm_state

    @classmethod
    def get_cache(
        cls,
        args,
        batch_size: int,
        max_seq_length: Optional[int]
    ) -> "Mamba2Cache":
        """Create a new cache instance with the given parameters."""
        return cls(
            batch_size=batch_size,
            intermediate_size=args.intermediate_size,
            state_size=args.state_size,
            conv_kernel=args.conv_kernel,
            num_heads=args.num_heads,
            head_dim=args.head_dim
        )