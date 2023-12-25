# Copyright © 2023 Apple Inc.

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Mistral or Llama models to MLX.",
    )
    parser.add_argument(
        "--torch-model",
        type=str,
        default="mistral-7B-v0.1/",
        help="The torch model directory",
    )
    parser.add_argument(
        "--mlx-model",
        type=str,
        default="mlx-mistral-7B-v0.1/",
        help="The directory to store the mlx model",
    )
    args = parser.parse_args()

    torch_path = Path(args.torch_model)
    if not os.path.exists(args.mlx_model):
        os.makedirs(args.mlx_model)
    mlx_path = Path(args.mlx_model)

    # Copy the tokenizer
    tokenizer_path = torch_path / "tokenizer.model"
    if not tokenizer_path.exists():
        print(f"Make sure there is a file tokenizer.model in {args.torch_model}")
        exit(0)
    shutil.copyfile(
        str(tokenizer_path),
        str(mlx_path / "tokenizer.model"),
    )

    # Copy the model weights
    state = torch.load(str(torch_path / "consolidated.00.pth"))
    np.savez(
        str(mlx_path / "weights.npz"),
        **{k: v.to(torch.float16).numpy() for k, v in state.items()},
    )

    # Copy the params
    with open(torch_path / "params.json", "r") as f:
        config = json.loads(f.read())
        unused = ["multiple_of"]
        for k in unused:
            if k in config:
                config.pop(k)
        n_heads = config["n_heads"]
        if "sliding_window" in config:
            config.pop("sliding_window")
        if "n_kv_heads" not in config:
            config["n_kv_heads"] = n_heads
        if "head_dim" not in config:
            config["head_dim"] = config["dim"] // n_heads
        if "hidden_dim" not in config:
            config["hidden_dim"] = state["layers.0.feed_forward.w1.weight"].shape[0]
    with open(mlx_path / "params.json", "w") as outfile:
        json.dump(config, outfile, indent=4)
