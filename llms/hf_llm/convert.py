# Copyright © 2023 Apple Inc.

import argparse
import collections
import copy
import glob
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
import transformers
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten, tree_map
from models import Model, ModelArgs


def fetch_from_hub(hf_path: str):
    model_path = snapshot_download(
        repo_id=hf_path,
        allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
    )
    weight_files = glob.glob(f"{model_path}/*.safetensors")
    if len(weight_files) == 0:
        raise FileNotFoundError("No weights found in {}".format(model_path))

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf).items())

    config = transformers.AutoConfig.from_pretrained(hf_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        hf_path,
    )
    return weights, config.to_dict(), tokenizer


def quantize(weights, config, args):
    quantized_config = copy.deepcopy(config)

    # Load the model:
    model = Model(ModelArgs.from_dict(config))
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


def upload_to_hub(path: str, name: str):
    from huggingface_hub import HfApi, logging

    logging.set_verbosity_info()

    api = HfApi()

    api.upload_folder(
        folder_path=path,
        repo_id=f"mlx-community/{name}",
        repo_type="model",
        multi_commits=True,
        multi_commits_verbose=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face model to MLX format"
    )
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
    parser.add_argument(
        "--upload-name",
        help="The name of model to upload to Hugging Face MLX Community",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    print("[INFO] Loading")
    weights, config, tokenizer = fetch_from_hub(args.hf_path)
    if args.quantize:
        print("[INFO] Quantizing")
        weights, config = quantize(weights, config, args)
    if not args.quantize:
        dtype = getattr(mx, args.dtype)
        weights = {k: v.astype(dtype) for k, v in weights.items()}

    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)
    shards = make_shards(weights)
    for i, shard in enumerate(shards):
        mx.save_safetensors(str(mlx_path / f"weights.{i:02d}.safetensors"), shard)
    tokenizer.save_pretrained(mlx_path)
    with open(mlx_path / "config.json", "w") as fid:
        json.dump(config, fid, indent=4)

    if args.upload_name is not None:
        upload_to_hub(mlx_path, args.upload_name)
