# Copyright © 2023 Apple Inc.

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch


def convert(k, v, config):
    v = v.to(torch.float16).numpy()
    if "block_sparse_moe" not in k:
        return [(k, v)]
    if "gate" in k:
        return [(k.replace("block_sparse_moe", "feed_forward"), v)]

    # From: layers.N.block_sparse_moe.w
    # To: layers.N.experts.M.w
    num_experts = args["moe"]["num_experts"]
    key_path = k.split(".")
    v = np.split(v, num_experts, axis=0)
    if key_path[-1] == "w2":
        v = [u.T for u in v]

    w_name = key_path.pop()
    key_path[-1] = "feed_forward.experts"
    return [
        (".".join(key_path + [str(e), w_name, "weight"]), u) for e, u in enumerate(v)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Mixtral weights to MLX.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Mixtral-8x7B-v0.1/",
        help="The path to the Mixtral model. The MLX model weights will also be saved there.",
    )
    args = parser.parse_args()
    model_path = Path(args.model_path)

    with open("params.json") as fid:
        args = json.load(fid)
        args["model_type"] = "mixtral"
    with open(model_path / "config.json", "w") as f:
        json.dump(args, f, indent=4)

    torch_files = glob.glob(str(model_path / "consolidated.*.pt"))
    torch_files = sorted(torch_files, key=lambda tf: int(tf.split(".")[-2]))
    for e, tf in enumerate(torch_files):
        print(f"[INFO] Converting file {e + 1}/{len(torch_files)}")
        state = torch.load(tf)
        new_state = {}
        for k, v in state.items():
            new_state.update(convert(k, v, args))
        np.savez(str(model_path / f"weights.{e}.npz"), **new_state)
