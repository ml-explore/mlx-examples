import argparse
import copy
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from phi2 import ModelArgs, Phi2
from transformers import AutoModelForCausalLM


def quantize(weights, config, args):
    quantized_config = copy.deepcopy(config)

    # Load the model:
    model = Phi2(ModelArgs())
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


def replace_key(key: str) -> str:
    if "wte.weight" in key:
        key = "wte.weight"

    if ".mlp" in key:
        key = key.replace(".mlp", "")
    return key


def convert():
    parser = argparse.ArgumentParser(description="Convert Phi-2 weights to MLX")
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

    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", torch_dtype="auto", trust_remote_code=True
    )
    state_dict = model.state_dict()
    weights = {replace_key(k): v.numpy() for k, v in state_dict.items()}
    params = {}
    if args.quantize:
        print("[INFO] Quantizing")
        weights, params = quantize(weights, params, args)

    np.savez(str(mlx_path / "weights.npz"), **weights)
    with open(mlx_path / "config.json", "w") as fid:
        params["model_type"] = "phi2"
        json.dump(params, fid, indent=4)


if __name__ == "__main__":
    convert()
