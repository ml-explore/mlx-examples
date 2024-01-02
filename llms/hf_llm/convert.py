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


def convert(hf_path: str, dtype: str):
    # Download model, config and tokenizer from HF
    model = transformers.AutoModelForCausalLM.from_pretrained(
            hf_path,
            trust_remote_code=True,
            torch_dtype=getattr(torch, dtype),
            ).state_dict()
    config = transformers.AutoConfig.from_pretrained(hf_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            hf_path,
            trust_remote_code=True,
    )

    # things to change
    # 1. there's no "model." in the weight names
    model = {k.replace("model.", ""): v for k, v in model.items()}

    # 2. mlp is called feed_forward
    model = {k.replace("mlp", "feed_forward"): v for k, v in model.items()}

    # 3. up_proj, down_proj, gate_proj
    model = {k.replace("down_proj", "w2"): v for k, v in model.items()}
    model = {k.replace("up_proj", "w3"): v for k, v in model.items()}
    model = {k.replace("gate_proj", "w1"): v for k, v in model.items()}

    # 4. layernorms
    model = {
        k.replace("input_layernorm", "attention_norm"): v for k, v in model.items()
    }
    model = {
        k.replace("post_attention_layernorm", "ffn_norm"): v for k, v in model.items()
    }

    # 5. lm head
    model = {k.replace("lm_head", "output"): v for k, v in model.items()}

    # 6. token emb
    model = {k.replace("embed_tokens", "tok_embeddings"): v for k, v in model.items()}

    # 7. attention
    model = {k.replace("self_attn", "attention"): v for k, v in model.items()}
    model = {k.replace("q_proj", "wq"): v for k, v in model.items()}
    model = {k.replace("k_proj", "wk"): v for k, v in model.items()}
    model = {k.replace("v_proj", "wv"): v for k, v in model.items()}
    model = {k.replace("o_proj", "wo"): v for k, v in model.items()}

    params = {}
    params["model_type"] = "llama"
    params["dim"] = config.hidden_size
    params["hidden_dim"] = config.intermediate_size
    params["head_dim"] = config.hidden_size // config.num_attention_heads
    params["n_heads"] = config.num_attention_heads
    if hasattr(config, "num_key_value_heads"):
        params["n_kv_heads"] = config.num_key_value_heads
    params["n_layers"] = config.num_hidden_layers
    params["vocab_size"] = config.vocab_size
    params["norm_eps"] = config.rms_norm_eps
    params["rope_traditional"] = False
    params["rope_theta"] = getattr(config, "rope_theta", 10000)

    for k, v in model.items():
        model[k] = mx.array(v.numpy())

    return model, params, tokenizer


def quantize(weights, config, args):
    quantized_config = copy.deepcopy(config)

    # Load the model:
    model = Model(ModelArgs(**config))
    weights = tree_map(mx.array, weights)
    # TODO replace with model.load_weights
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
    weights, config, tokenizer = convert(args.hf_path, args.dtype)
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
