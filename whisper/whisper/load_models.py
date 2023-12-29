# Copyright © 2023 Apple Inc.

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from . import whisper


def load_model(
    folder: str,
    dtype: mx.Dtype = mx.float32,
) -> whisper.Whisper:
    model_path = Path(folder)

    with open(str(model_path / "config.json"), "r") as f:
        config = json.loads(f.read())
        config.pop("model_type", None)
        quantization = config.pop("quantization", None)

    model_args = whisper.ModelDimensions(**config)

    weights = mx.load(str(model_path / "weights.npz"))
    weights = tree_unflatten(list(weights.items()))

    model = whisper.Whisper(model_args, dtype)

    if quantization is not None:
        nn.QuantizedLinear.quantize_module(model, **quantization)

    model.update(weights)
    mx.eval(model.parameters())
    return model
