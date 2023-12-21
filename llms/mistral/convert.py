# Copyright © 2023 Apple Inc.

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mistral import Mistral, ModelArgs
from mlx.utils import tree_flatten, tree_map, tree_unflatten


def quantize(weights, config):
    quantized_config = copy.deepcopy(config)

    # Load the model:
    config.pop("sliding_window", None)
    model = Mistral(ModelArgs(**config))
    weights = tree_map(mx.array, weights)
    model.update(tree_unflatten(list(weights.items())))

    # Quantize the model:
    nn.QuantizedLinear.quantize_module(model)

    # Update the config:
    quantized_config["quantization"] = {"group_size": 64, "bits": 4}
    quantized_weights = dict(tree_flatten(model.parameters()))

    return quantized_weights, quantized_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Mistral weights to MLX.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="mistral-7B-v0.1/",
        help="The path to the Mistral model. The MLX weights will also be saved there.",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        help="Generate a 4-bit quantized model.",
        action="store_true",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    state = torch.load(str(model_path / "consolidated.00.pth"))
    weights = {k: v.to(torch.float16).numpy() for k, v in state.items()}
    if args.quantize:
        print("[INFO] Quantizing")
        weights, params = quantize(weights, params)

    np.savez(str(model_path / "weights.npz"), **weights)

    # Save config.json with model_type
    with open(model_path / "params.json", "r") as f:
        config = json.loads(f.read())
        config["model_type"] = "mistral"
    with open(model_path / "config.json", "w") as f:
        json.dump(config, f, indent=4)
