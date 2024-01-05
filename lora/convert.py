# Copyright © 2023 Apple Inc.

import argparse
import copy
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from lora import Model, ModelArgs


def quantize(weights, config, args):
    quantized_config = copy.deepcopy(config)

    # Load the model:
    model = Model(ModelArgs(**config))
    weights = tree_map(mx.array, weights)
    model.update(tree_unflatten(list(weights.items())))

    # Quantize the model:
    nn.QuantizedLinear.quantize_module(
        model,
        args.q_group_size,
        args.q_bits,
        linear_class_predicate=lambda m: isinstance(m, nn.Linear)
        and m.weight.shape[0] != config["vocab_size"],
    )

    # Update the config:
    quantized_config["quantization"] = {
        "group_size": args.q_group_size,
        "bits": args.q_bits,
    }
    quantized_weights = dict(tree_flatten(model.parameters()))

    return quantized_weights, quantized_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Mistral or Llama models to MLX.",
    )
    parser.add_argument(
        "--torch-path",
        type=str,
        default="mistral-7B-v0.1/",
        help="Path to the torch model directory",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx_model/",
        help="The directory to store the mlx model",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        help="Generate a quantized model.",
        action="store_true",
    )
    parser.add_argument(
        "--q-group-size",
        help="Group size for quantization.",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--q-bits",
        help="Bits per weight for quantization.",
        type=int,
        default=4,
    )
    args = parser.parse_args()

    args = parser.parse_args()

    torch_path = Path(args.torch_path)
    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    # Copy the tokenizer
    tokenizer_path = torch_path / "tokenizer.model"
    if not tokenizer_path.exists():
        print(f"Make sure there is a file tokenizer.model in {args.torch-path}")
        exit(0)
    shutil.copyfile(
        str(tokenizer_path),
        str(mlx_path / "tokenizer.model"),
    )

    # Load the torch model weights to numpy:
    weights = torch.load(str(torch_path / "consolidated.00.pth"))
    for k, v in weights.items():
        weights[k] = v.to(torch.float16).numpy()

    # Standardize the params
    with open(torch_path / "params.json", "r") as f:
        config = json.loads(f.read())
        unused = ["multiple_of", "sliding_window"]
        for k in unused:
            config.pop(k, None)
        n_heads = config["n_heads"]
        if "n_kv_heads" not in config:
            config["n_kv_heads"] = n_heads
        if "head_dim" not in config:
            config["head_dim"] = config["dim"] // n_heads
        if "hidden_dim" not in config:
            config["hidden_dim"] = weights["layers.0.feed_forward.w1.weight"].shape[0]
        if config.get("vocab_size", -1) < 0:
            config["vocab_size"] = weights["output.weight"].shape[0]

    if args.quantize:
        print("[INFO] Quantizing")
        weights, config = quantize(weights, config, args)

    np.savez(str(mlx_path / "weights.npz"), **weights)

    with open(mlx_path / "config.json", "w") as outfile:
        json.dump(config, outfile, indent=4)
