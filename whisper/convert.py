# Copyright © 2023 Apple Inc.

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from whisper.load_models import load_torch_model, torch_to_mlx

MODEL_DTYPES = {"float16", "float32"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Mistral weights to MLX.")
    parser.add_argument(
        "--torch-name-or-path",
        type=str,
        default="tiny",
        help="The name or path to the PyTorch model.",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx_model",
        help="The path to save the MLX model.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="The dtype to save the MLX model.",
    )
    args = parser.parse_args()

    assert args.dtype in MODEL_DTYPES, f"dtype {args.dtype} not found in {MODEL_DTYPES}"
    dtype = getattr(mx, args.dtype)

    model = torch_to_mlx(load_torch_model(args.torch_name_or_path), dtype)
    config = asdict(model.dims)
    weights = dict(tree_flatten(model.parameters()))

    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    # Save weights
    np.savez(str(mlx_path / "weights.npz"), **weights)

    # Save config.json with model_type
    with open(mlx_path / "config.json", "w") as f:
        config["model_type"] = "whisper"
        json.dump(config, f, indent=4)
