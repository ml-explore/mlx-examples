# Copyright © 2024 Gökdeniz Gülmez

from pathlib import Path
from typing import Union

import mlx.core as mx
import mlx.nn as nn

from mlx.utils import tree_flatten

def print_trainable_parameters(model):
    def nparams(m):
        if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            return m.weight.size * (32 // m.bits)
        return sum(v.size for _, v in tree_flatten(m.parameters()))

    leaf_modules = tree_flatten(
        model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)
    )
    total_p = sum(nparams(m) for _, m in leaf_modules) / 10**6
    trainable_p = (
        sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    )
    print(
        f"Training model with: {trainable_p:.3f}M) Params"
    )

def save_model(model: nn.Module, save_path):
    flattened_tree = tree_flatten(model.trainable_parameters())
    mx.save_safetensors(str(save_path), dict(flattened_tree))
    print(f"Saved model to {str(save_path)}")