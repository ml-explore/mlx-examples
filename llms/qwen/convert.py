import argparse
import copy
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from qwen import ModelArgs, Qwen
from transformers import AutoModelForCausalLM


def replace_key(key: str) -> str:
    if key.startswith("transformer."):
        # remove transformer prefix
        key = key.replace("transformer.", "")

    return key


def quantize(weights, config, args):
    quantized_config = copy.deepcopy(config)

    # Load the model:
    model_args = ModelArgs()
    model_args.vocab_size = config["vocab_size"]
    model_args.hidden_size = config["hidden_size"]
    model_args.num_attention_heads = config["num_attention_heads"]
    model_args.num_hidden_layers = config["num_hidden_layers"]
    model_args.kv_channels = config["kv_channels"]
    model_args.max_position_embeddings = config["max_position_embeddings"]
    model_args.layer_norm_epsilon = config["layer_norm_epsilon"]
    model_args.intermediate_size = config["intermediate_size"]
    model_args.no_bias = config["no_bias"]
    model = Qwen(model_args)

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


def convert(args):
    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float16
    )
    state_dict = model.state_dict()
    weights = {replace_key(k): v.numpy() for k, v in state_dict.items()}
    config = model.config.to_dict()

    if args.quantize:
        print("[INFO] Quantizing")
        weights, config = quantize(weights, config, args)

    np.savez(str(mlx_path / "weights.npz"), **weights)

    # write config
    with open(mlx_path / "config.json", "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Qwen model to npz")

    parser.add_argument(
        "--model",
        help="The huggingface model to be converted",
        default="Qwen/Qwen-1_8B",
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
    convert(args)
