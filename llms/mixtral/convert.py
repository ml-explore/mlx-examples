# Copyright © 2023 Apple Inc.

import argparse
import copy
import glob
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mixtral import Mixtral, ModelArgs
from mlx.utils import tree_flatten, tree_map, tree_unflatten


def convert(tf, config):
    def convert_single(k, v):
        v = v.to(torch.float16).numpy()
        if "block_sparse_moe" not in k:
            return [(k, v)]
        if "gate" in k:
            return [(k.replace("block_sparse_moe", "feed_forward"), v)]

        # From: layers.N.block_sparse_moe.w
        # To: layers.N.experts.M.w
        num_experts = config["moe"]["num_experts"]
        key_path = k.split(".")
        v = np.split(v, num_experts, axis=0)
        if key_path[-1] == "w2":
            v = [u.T for u in v]

        w_name = key_path.pop()
        key_path[-1] = "feed_forward.experts"
        return [
            (".".join(key_path + [str(e), w_name, "weight"]), u)
            for e, u in enumerate(v)
        ]

    state = torch.load(tf)
    weights = {}
    for k, v in state.items():
        weights.update(convert_single(k, v))
    return weights


def quantize(weights, config, args):
    quantized_config = copy.deepcopy(config)

    # Load the model and update with the subset of weights:
    config.pop("quantization", None)
    model = Mixtral(ModelArgs(**config))
    all_weights = dict(tree_flatten(model.parameters()))

    weights = tree_map(mx.array, weights)

    all_weights.update(weights)
    all_weights = tree_unflatten(list(all_weights.items()))
    model.update(all_weights)

    # Quantize the model:
    nn.QuantizedLinear.quantize_module(
        model,
        args.q_group_size,
        args.q_bits,
        # TODO: Quantize gate matrices when < 32 tiles supported
        linear_class_predicate=lambda m: isinstance(m, nn.Linear)
        and m.weight.shape[0] != 8,
    )

    # Extract the subset of quantized weights:
    all_weights = dict(tree_flatten(model.parameters()))
    quantized_weights = {}
    for k, v in all_weights.items():
        if k not in weights:
            continue
        quantized_weights[k] = v
        prefix = k.split(".")[:-1]
        for qw in ["scales", "biases"]:
            if (k := ".".join(prefix + [qw])) in all_weights:
                quantized_weights[k] = all_weights[k]

    # Update the config:
    quantized_config["quantization"] = {
        "group_size": args.q_group_size,
        "bits": args.q_bits,
    }
    return quantized_weights, quantized_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Mixtral weights to MLX.")
    parser.add_argument(
        "--torch-path",
        type=str,
        default="Mixtral-8x7B-v0.1",
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
    torch_path = Path(args.torch_path)
    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    with open("params.json") as fid:
        config = json.load(fid)

    # Copy tokenizer
    shutil.copyfile(
        str(torch_path / "tokenizer.model"),
        str(mlx_path / "tokenizer.model"),
    )

    # Convert and save model in shards
    torch_files = glob.glob(str(torch_path / "consolidated.*.pt"))
    torch_files = sorted(torch_files, key=lambda tf: int(tf.split(".")[-2]))
    for e, tf in enumerate(torch_files):
        print(f"[INFO] Converting file {e + 1}/{len(torch_files)}")
        weights = convert(tf, config)
        if args.quantize:
            print("[INFO] Quantizing")
            weights, config = quantize(weights, config, args)
        np.savez(str(mlx_path / f"weights.{e}.npz"), **weights)

    # Save updated config
    with open(mlx_path / "config.json", "w") as f:
        config["model_type"] = "mixtral"
        json.dump(config, f, indent=4)
