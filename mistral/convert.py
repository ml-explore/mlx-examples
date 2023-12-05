# Copyright © 2023 Apple Inc.

import argparse
import numpy as np
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Mistral weights to MLX.")
    parser.add_argument(
        "--torch_model",
        type=str,
        default="mistral-7B-v0.1/consolidated.00.pth",
        help="The path to the torch model weights",
    )
    parser.add_argument(
        "--mlx_model",
        type=str,
        default="mistral-7B-v0.1/mlx_mistral_7b.npz",
        help="The path to store the mlx model weights",
    )
    args = parser.parse_args()

    state = torch.load(args.torch_model)
    np.savez(
        args.mlx_model, **{k: v.to(torch.float16).numpy() for k, v in state.items()}
    )
