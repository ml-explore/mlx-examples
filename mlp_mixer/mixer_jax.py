# Adapted from https://raw.githubusercontent.com/google-research/vision_transformer/main/vit_jax/models_mixer.py.
# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
from typing import Dict, Optional, Tuple

import einops
import flax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from config import JAX_WEIGHTS_PATH, MODELS
from flax.core.frozen_dict import FrozenDict
from flax.training import checkpoints


class MlpBlock(nn.Module):
    mlp_dim: int

    @nn.compact
    def __call__(self, x):
        y = nn.Dense(self.mlp_dim)(x)
        y = nn.gelu(y)
        return nn.Dense(x.shape[-1])(y)


class MixerBlock(nn.Module):
    """Mixer block layer."""

    tokens_mlp_dim: int
    channels_mlp_dim: int

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm()(x)
        y = jnp.swapaxes(y, 1, 2)
        y = MlpBlock(self.tokens_mlp_dim, name="token_mixing")(y)
        y = jnp.swapaxes(y, 1, 2)
        x = x + y
        y = nn.LayerNorm()(x)
        return x + MlpBlock(self.channels_mlp_dim, name="channel_mixing")(y)


class MlpMixer(nn.Module):
    """Mixer architecture."""

    patch_size: int
    num_classes: int
    num_blocks: int
    hidden_dim: int
    tokens_mlp_dim: int
    channels_mlp_dim: int
    model_name: Optional[str] = None

    @nn.compact
    def __call__(self, inputs, *, train):
        del train
        x = nn.Conv(
            self.hidden_dim,
            (self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            name="stem",
        )(inputs)
        x = einops.rearrange(x, "n h w c -> n (h w) c")
        for _ in range(self.num_blocks):
            x = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)(x)
        x = nn.LayerNorm(name="pre_head_layer_norm")(x)
        x = jnp.mean(x, axis=1)
        if self.num_classes:
            x = nn.Dense(
                self.num_classes, kernel_init=nn.initializers.zeros, name="head"
            )(x)
        return x


def recover_tree(keys, values):
    """Recovers a tree as a nested dict from flat names and values.

    This function is useful to analyze checkpoints that are without need to access
    the exact source code of the experiment. In particular, it can be used to
    extract an reuse various subtrees of the scheckpoint, e.g. subtree of
    parameters.

    Args:
      keys: a list of keys, where '/' is used as separator between nodes.
      values: a list of leaf values.

    Returns:
      A nested tree-like dict.
    """
    tree = {}
    sub_trees = collections.defaultdict(list)
    for k, v in zip(keys, values):
        if "/" not in k:
            tree[k] = v
        else:
            k_left, k_right = k.split("/", 1)
            sub_trees[k_left].append((k_right, v))
    for k, kv_pairs in sub_trees.items():
        k_subtree, v_subtree = zip(*kv_pairs)
        tree[k] = recover_tree(k_subtree, v_subtree)
    return tree


def load_jax_weights(path: str) -> Dict[str, jnp.array]:
    """Loads params from a checkpoint previously stored with `save()`."""
    ckpt_dict = np.load(path, allow_pickle=False)
    keys, values = zip(*list(ckpt_dict.items()))
    params = checkpoints.convert_pre_linen(recover_tree(keys, values))
    if isinstance(params, flax.core.FrozenDict):
        params = params.unfreeze()
    return params


def load(
    model_name: str, jax_weights_path: str = JAX_WEIGHTS_PATH
) -> Tuple[MlpMixer, FrozenDict]:
    assert model_name in MODELS

    config = MODELS[model_name]["config"]
    model = MlpMixer(
        patch_size=config.patch_size,
        num_classes=config.num_classes,
        num_blocks=config.num_blocks,
        hidden_dim=config.hidden_dim,
        tokens_mlp_dim=config.tokens_mlp_dim,
        channels_mlp_dim=config.channels_mlp_dim,
    )

    variables = FrozenDict(
        {"params": load_jax_weights(f"{jax_weights_path}/{model_name}.npz")}
    )

    return model, variables
