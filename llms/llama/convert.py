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
import torch
from llama import Llama, ModelArgs, sanitize_config
from mlx.utils import tree_flatten, tree_map, tree_unflatten


def torch_to_mx(a: torch.Tensor, *, dtype: str) -> mx.array:
    # bfloat16 is not numpy convertible. Upcast to float32 to avoid precision loss
    a = a.to(torch.float32) if dtype == "bfloat16" else a.to(getattr(torch, dtype))
    return mx.array(a.numpy(), getattr(mx, dtype))


def llama(model_path, *, dtype: str):
    SHARD_FIRST = ["wv", "wq", "wk", "w1", "w3", "output"]
    SHARD_SECOND = ["tok_embeddings", "wo", "w2"]
    SHARD_WEIGHTS = set(SHARD_FIRST + SHARD_SECOND)

    def shard_key(k):
        keys = k.split(".")
        if len(keys) < 2:
            return None
        return keys[-2]

    def unshard(k, v):
        wn = shard_key(k)
        if wn not in SHARD_WEIGHTS:
            return v
        elif wn in SHARD_FIRST:
            axis = 0
        elif wn in SHARD_SECOND:
            axis = 1
        else:
            raise ValueError("Invalid weight name")
        return mx.concatenate(v, axis=axis)

    torch_files = glob.glob(str(model_path / "consolidated.*.pth"))
    weights = collections.defaultdict(list)
    for wf in torch_files:
        state = torch.load(wf, map_location=torch.device("cpu"))
        for k, v in state.items():
            v = torch_to_mx(v, dtype=dtype)
            state[k] = None  # free memory
            if shard_key(k) in SHARD_WEIGHTS:
                weights[k].append(v)
            else:
                weights[k] = v

    for k, v in weights.items():
        weights[k] = unshard(k, v)
    with open(model_path / "params.json", "r") as f:
        params = json.loads(f.read())
    return weights, params


def tiny_llama(model_path, *, dtype: str):
    try:
        import transformers
    except ImportError:
        print("The transformers package must be installed for this model conversion:")
        print("pip install transformers")
        exit(1)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_path)
    ).state_dict()
    config = transformers.AutoConfig.from_pretrained(model_path)

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
    params["dim"] = config.hidden_size
    params["hidden_dim"] = config.intermediate_size
    params["n_heads"] = config.num_attention_heads
    if hasattr(config, "num_key_value_heads"):
        params["n_kv_heads"] = config.num_key_value_heads
    params["n_layers"] = config.num_hidden_layers
    params["vocab_size"] = config.vocab_size
    params["norm_eps"] = config.rms_norm_eps
    params["rope_traditional"] = False
    weights = {k: torch_to_mx(v, dtype=dtype) for k, v in model.items()}

    return weights, params


def quantize(weights, config, args):
    quantized_config = copy.deepcopy(config)

    # Load the model:
    config = sanitize_config(config, weights)
    model = Llama(ModelArgs(**config))
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


def make_shards(weights: dict, max_file_size_gibibyte: int = 15):
    max_file_size_bytes = max_file_size_gibibyte << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Llama weights to MLX")
    parser.add_argument(
        "--torch-path",
        type=str,
        help="Path to the PyTorch model.",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx_model",
        help="Path to save the MLX model.",
    )
    parser.add_argument(
        "--model-name",
        help=(
            "Name of the model to convert. Use 'llama' for models in the "
            "Llama family distributed by Meta including Llama 1, Llama 2, "
            "Code Llama, and Llama chat."
        ),
        choices=["tiny_llama", "llama"],
        default="llama",
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
        help="dtype for loading the torch model and input for quantization or saving the converted model. "
        "The original weights are stored in bfloat16.",
        type=str,
        default="float16",
    )

    args = parser.parse_args()

    torch_path = Path(args.torch_path)
    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading")
    weights, params = globals()[args.model_name](torch_path, dtype=args.dtype)
    params["model_type"] = "llama"
    if args.quantize:
        print("[INFO] Quantizing")
        weights, params = quantize(weights, params, args)

    print("[INFO] Saving")
    shutil.copyfile(
        str(torch_path / "tokenizer.model"),
        str(mlx_path / "tokenizer.model"),
    )
    shards = make_shards(weights)
    if len(shards) == 1:
        mx.savez(str(mlx_path / f"weights.npz"), **shards[0])
    else:
        for i, shard in enumerate(shards):
            mx.savez(str(mlx_path / f"weights.{i:02d}.npz"), **shard)
    with open(mlx_path / "config.json", "w") as fid:
        json.dump(params, fid, indent=4)
