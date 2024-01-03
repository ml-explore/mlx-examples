# Copyright © 2023 Apple Inc.

import argparse
import collections
import copy
import glob
import json
import shutil
from pathlib import Path
import transformers

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from models import Model, ModelArgs
from mlx.utils import tree_flatten, tree_map, tree_unflatten


def fetch_from_hub(hf_path: str, dtype: str):
    model = transformers.AutoModelForCausalLM.from_pretrained(
            hf_path,
            torch_dtype=getattr(torch, dtype),
            ).state_dict()
    config = transformers.AutoConfig.from_pretrained(hf_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            hf_path,
    )
    for k, v in model.items():
        model[k] = mx.array(v.numpy())
    return model, config.to_dict(), tokenizer


def quantize(weights, config, args):
    quantized_config = copy.deepcopy(config)

    # Load the model:
    model = Model(ModelArgs(**config))
    weights = tree_map(mx.array, weights)
    model.load_weights(list(weights.items()))

    # Quantize the model:
    nn.QuantizedLinear.quantize_module(model, args.q_group_size, args.q_bits)

    # Update the config:
    quantized_config["quantization"] = {
        "group_size": args.q_group_size,
        "bits": args.q_bits,
    }
    quantized_weights = dict(tree_flatten(model.parameters()))

    return quantized_weights, quantized_config


def make_shards(weights: dict, max_file_size_gibibyte: int = 15):
    max_file_size_bytes = max_file_size_gibibyte << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        estimated_size = v.size * v.dtype.size
        if shard_size + estimated_size > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += estimated_size
    shards.append(shard)
    return shards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face model to MLX format")
    parser.add_argument(
        "--hf-path",
        type=str,
        help="Path to the Hugging Face model.",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx_model",
        help="Path to save the MLX model.",
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
    parser.add_argument(
        "--dtype",
        help="Type to save the parameters, ignored if -q is given.",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )

    args = parser.parse_args()

    print("[INFO] Loading")
    weights, config, tokenizer = fetch_from_hub(args.hf_path, args.dtype)
    if args.quantize:
        print("[INFO] Quantizing")
        weights, config = quantize(weights, config, args)

    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)
    shards = make_shards(weights)
    for i, shard in enumerate(shards):
        mx.save_safetensors(str(mlx_path / f"weights.{i:02d}.safetensors"), shard)
    tokenizer.save_pretrained(mlx_path)
    with open(mlx_path / "config.json", "w") as fid:
        json.dump(config, fid, indent=4)
