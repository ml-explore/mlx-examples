# Copyright © 2023 Apple Inc.

import argparse
import collections
import glob
import json
import numpy as np
from pathlib import Path
import torch


def llama(model_path):
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
        return np.concatenate(v, axis=axis)

    torch_files = glob.glob(str(model_path / "consolidated.*.pth"))
    weights = collections.defaultdict(list)
    for wf in torch_files:
        state = torch.load(wf, map_location=torch.device("cpu"))
        for k, v in state.items():
            v = v.to(torch.float16).numpy()
            if shard_key(k) in SHARD_WEIGHTS:
                weights[k].append(v)
            else:
                weights[k] = v

    for k, v in weights.items():
        weights[k] = unshard(k, v)
    return weights, None


def tiny_llama(model_path):
    try:
        import transformers
    except ImportError as e:
        print("The transformers package must be installed for this model conversion:")
        print("pip install transformers")
        import sys

        sys.exit(0)

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
    weights = {k: v.to(torch.float16).numpy() for k, v in model.items()}

    return weights, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Llama weights to MLX")
    parser.add_argument(
        "--model-path",
        help="Path to the model. The MLX weights will also be saved there.",
    )
    parser.add_argument(
        "--model-name",
        help=(
            "Name of the model to convert. Use 'llama' for models in the "
            "Llama family distributed by Meta including Llama 1, Llama 2, "
            "Coda Llama, and Llama chat."
        ),
        choices=["tiny_llama", "llama"],
        default="llama",
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)
    weights, params = globals()[args.model_name](model_path)
    np.savez(str(model_path / "weights.npz"), **weights)
    if params is not None:
        with open(model_path / "params.json", "w") as fid:
            json.dump(params, fid, indent=4)
