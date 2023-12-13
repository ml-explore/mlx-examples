# Copyright © 2023 Apple Inc.

import argparse
import collections
import glob
from pathlib import Path

import numpy as np
import torch

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Llama weights to MLX")
    parser.add_argument(
        "--model_path",
        help="Path to the Torch model. The MLX weights will also be saved there.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
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

    out_file = str(model_path / "weights.npz")
    for k, v in weights.items():
        weights[k] = unshard(k, v)
    np.savez(out_file, **weights)
