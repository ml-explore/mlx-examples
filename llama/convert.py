# Copyright © 2023 Apple Inc.

import argparse
from itertools import starmap

import numpy as np
import torch


def map_torch_to_mlx(key, value):
    if "tok_embedding" in key:
        key = "embedding.weight"

    elif "norm" in key:
        key = key.replace("attention_norm", "norm1").replace("ffn_norm", "norm2")

    elif "wq" in key or "wk" in key or "wv" in key or "wo" in key:
        key = key.replace("wq", "query_proj")
        key = key.replace("wk", "key_proj")
        key = key.replace("wv", "value_proj")
        key = key.replace("wo", "out_proj")

    elif "w1" in key or "w2" in key or "w3" in key:
        # The FFN is a separate submodule in PyTorch
        key = key.replace("feed_forward.w1", "linear1")
        key = key.replace("feed_forward.w3", "linear2")
        key = key.replace("feed_forward.w2", "linear3")

    elif "output" in key:
        key = key.replace("output", "out_proj")

    elif "rope" in key:
        return None, None

    return key, value.numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Llama weights to MLX")
    parser.add_argument("torch_weights")
    parser.add_argument("output_file")
    args = parser.parse_args()

    state = torch.load(args.torch_weights)
    np.savez(
        args.output_file,
        **{k: v for k, v in starmap(map_torch_to_mlx, state.items()) if k is not None}
    )
