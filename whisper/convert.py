# Copyright © 2023 Apple Inc.

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from whisper.load_models import load_torch_model, torch_to_mlx
from whisper.whisper import ModelDimensions, Whisper

MODEL_DTYPES = {"float16", "float32"}


def quantize(weights, config, args):
    quantized_config = copy.deepcopy(config)

    # Load the model:
    model = Whisper(ModelDimensions(**config))
    weights = tree_map(mx.array, weights)
    model.update(tree_unflatten(list(weights.items())))

    # Quantize the model:
    nn.QuantizedLinear.quantize_module(model, args.q_group_size, args.q_bits)

    # Update the config:
    quantized_config["quantization"] = {
        "group_size": args.q_group_size,
        "bits": args.q_bits,
    }
    quantized_weights = dict(tree_flatten(model.parameters()))

    return quantized_weights, quantized_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Whisper weights to MLX.")
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
    parser.add_argument(
        "-q",
        "--quantize",
        help="Generate a quantized model.",
        action="store_true",
    )
    parser.add_argument(
        "--q_group_size",
        help="Group size for quantization.",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--q_bits",
        help="Bits per weight for quantization.",
        type=int,
        default=4,
    )
    args = parser.parse_args()

    assert args.dtype in MODEL_DTYPES, f"dtype {args.dtype} not found in {MODEL_DTYPES}"
    dtype = getattr(mx, args.dtype)

    print("[INFO] Loading")
    model = torch_to_mlx(load_torch_model(args.torch_name_or_path), dtype)
    config = asdict(model.dims)
    weights = dict(tree_flatten(model.parameters()))

    if args.quantize:
        print("[INFO] Quantizing")
        weights, config = quantize(weights, config, args)

    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    # Save weights
    print("[INFO] Saving")
    np.savez(str(mlx_path / "weights.npz"), **weights)

    # Save config.json with model_type
    with open(str(mlx_path / "config.json"), "w") as f:
        config["model_type"] = "whisper"
        json.dump(config, f, indent=4)
