import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def replace_key(key: str) -> str:
    if key.startswith("transformer."):
        # remove transformer prefix
        key = key.replace("transformer.", "")

    return key


def convert(model_path: str = "Qwen/Qwen-1_8B"):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float16
    )
    state_dict = model.state_dict()
    weights = {replace_key(k): v.numpy() for k, v in state_dict.items()}
    np.savez("weights.npz", **weights)

    # write config
    config = model.config
    config_dict = config.to_dict()
    with open("config.json", "w") as f:
        json.dump(config_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Qwen model to npz")

    parser.add_argument(
        "--model",
        help="The huggingface model to be converted",
        default="Qwen/Qwen-1_8B",
    )

    args = parser.parse_args()

    convert(args.model)
