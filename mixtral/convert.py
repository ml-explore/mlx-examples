# Copyright © 2023 Apple Inc.

import argparse
import numpy as np
from pathlib import Path
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Mixtral weights to MLX.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="mixtral-8x7b-32kseqlen/",
        help="The path to the Mixtral model. The MLX model weights will also be saved there.",
    )
    args = parser.parse_args()
    model_path = Path(args.model_path)
    state = torch.load(str(model_path / "consolidated.00.pth"))
    np.savez(
        str(model_path / "weights.npz"),
        **{k: v.to(torch.float16).numpy() for k, v in state.items()},
    )
