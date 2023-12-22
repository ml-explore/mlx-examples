import argparse
from pathlib import Path
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    args = parser.parse_args()
    convert(args)
