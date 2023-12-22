import argparse
import copy
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepseek_coder import ModelArgs, DeepseekCoder


def quantize(weights, config, args):
    quantized_config = copy.deepcopy(config)

    # Load the model:
    model_args = ModelArgs()
    model_args.vocab_size = config["vocab_size"]
    model_args.hidden_size = config["hidden_size"]
    model_args.num_attention_heads = config["num_attention_heads"]
    model_args.num_key_value_heads = config["num_key_value_heads"]
    model_args.num_hidden_layers = config["num_hidden_layers"]
    model_args.max_position_embeddings = config["max_position_embeddings"]
    model_args.rms_norm_eps = config["rms_norm_eps"]
    model_args.intermediate_size = config["intermediate_size"]
    model_args.rope_scaling_factor = config["rope_scaling"]["factor"]
    model = DeepseekCoder(model_args)

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
    model_path = Path(args.model_path)

    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), trust_remote_code=True, torch_dtype=torch.float16
    )
    config = model.config.to_dict()

    state_dict = model.state_dict()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    # things to change
    # 1. there's no "model." in the weight names
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    # 2. mlp is called feed_forward
    state_dict = {k.replace("mlp", "feed_forward"): v for k, v in state_dict.items()}

    # 3. up_proj, down_proj, gate_proj
    state_dict = {k.replace("down_proj", "w2"): v for k, v in state_dict.items()}
    state_dict = {k.replace("up_proj", "w3"): v for k, v in state_dict.items()}
    state_dict = {k.replace("gate_proj", "w1"): v for k, v in state_dict.items()}

    # 4. layernorms
    state_dict = {
        k.replace("input_layernorm", "attention_norm"): v for k, v in state_dict.items()
    }
    state_dict = {
        k.replace("post_attention_layernorm", "ffn_norm"): v
        for k, v in state_dict.items()
    }

    # 5. lm head
    state_dict = {k.replace("lm_head", "output"): v for k, v in state_dict.items()}

    # 6. token emb
    state_dict = {
        k.replace("embed_tokens", "tok_embeddings"): v for k, v in state_dict.items()
    }

    # 7. attention
    state_dict = {k.replace("self_attn", "attention"): v for k, v in state_dict.items()}
    state_dict = {k.replace("q_proj", "wq"): v for k, v in state_dict.items()}
    state_dict = {k.replace("k_proj", "wk"): v for k, v in state_dict.items()}
    state_dict = {k.replace("v_proj", "wv"): v for k, v in state_dict.items()}
    state_dict = {k.replace("o_proj", "wo"): v for k, v in state_dict.items()}

    weights = {k: v.numpy() for k, v in state_dict.items()}

    if args.quantize:
        print("[INFO] Quantizing")
        weights, config = quantize(weights, config, args)

    np.savez(str(mlx_path / "weights.npz"), **weights)
    tokenizer.save_pretrained(mlx_path)
    with open(mlx_path / "config.json", "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Deepseek coder model to npz")

    parser.add_argument(
        "--model-path",
        help="The huggingface model to be converted",
        default="deepseek-ai/deepseek-coder-6.7b-instruct",
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
