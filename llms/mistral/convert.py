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
from mistral import Mistral, ModelArgs
from mlx.utils import tree_flatten, tree_map, tree_unflatten


def savez_stream(filename, **kwargs):
    import os
    import zipfile
    with zipfile.ZipFile(filename, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for k, v in kwargs.items():
            print(f"EVALING AND SAVING {k}")
            mx.eval(v)
            tmpfilename = f"{k}.npy"
            np.save(tmpfilename, v)
            del v
            zf.write(tmpfilename)
            os.remove(tmpfilename)


def quantize(weights, config, args):
    quantized_config = copy.deepcopy(config)

    # Load the model:
    config.pop("sliding_window", None)
    model = Mistral(ModelArgs(**config))
    for k, v in weights.items():
        weights[k] = mx.array(v)
        del v

    model.update(tree_unflatten(list(weights.items())))
    del weights

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
    parser = argparse.ArgumentParser(description="Convert Mistral weights to MLX.")
    parser.add_argument(
        "--torch-path",
        type=str,
        default="mistral-7B-v0.1",
        help="The path to the PyTorch model.",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx_model",
        help="The path to save the MLX model.",
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

    torch_path = Path(args.torch_path)
    weights = torch.load(str(torch_path / "consolidated.00.pth"), mmap=True)
    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    weights = {k: v.to(torch.float16).numpy() for k, v in weights.items()}
    with open(torch_path / "params.json", "r") as f:
        config = json.loads(f.read())

    if args.quantize:
        print("[INFO] Quantizing")
        weights, config = quantize(weights, config, args)

    # Save weights
    savez_stream(str(mlx_path / "weights.npz"), **weights)

    # Copy tokenizer
    shutil.copyfile(
        str(torch_path / "tokenizer.model"),
        str(mlx_path / "tokenizer.model"),
    )

    # Save config.json with model_type
    with open(mlx_path / "config.json", "w") as f:
        config["model_type"] = "mistral"
        json.dump(config, f, indent=4)
