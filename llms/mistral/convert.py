# Copyright © 2023 Apple Inc.

import argparse
import json
from pathlib import Path

import numpy as np
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Mistral weights to MLX.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="mistral-7B-v0.1/",
        help="The path to the Mistral model. The MLX weights will also be saved there.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    state = torch.load(str(model_path / "consolidated.00.pth"))
    np.savez(
        str(model_path / "weights.npz"),
        **{k: v.to(torch.float16).numpy() for k, v in state.items()}
    )

    # Save config.json with model_type
    with open(model_path / "params.json", "r") as f:
        config = json.loads(f.read())
        config["model_type"] = "mistral"
    with open(model_path / "config.json", "w") as f:
        json.dump(config, f, indent=4)
