import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import mlx.core as mx
import torch


def torch_to_mx(a: torch.Tensor, *, dtype: str) -> mx.array:
    # bfloat16 is not numpy convertible. Upcast to float32 to avoid precision loss
    a = a.to(torch.float32) if dtype == "bfloat16" else a.to(getattr(torch, dtype))
    return mx.array(a.numpy(), getattr(mx, dtype))


def map_weights(key: str, value: torch.Tensor) -> Tuple[str, mx.array]:
    key = key.replace("embeddings.", "")
    key = key.replace("encoder.", "")
    key = key.replace("position_embedding.weight", "position_embedding")

    # Map attention layers
    if "self_attn." in key:
        key = key.replace("self_attn.", "attention.")
    if "q_proj." in key:
        key = key.replace("q_proj.", "query_proj.")
    if "k_proj." in key:
        key = key.replace("k_proj.", "key_proj.")
    if "v_proj." in key:
        key = key.replace("v_proj.", "value_proj.")
    if "layer_norm1." in key:
        key = key.replace("layer_norm1.", "ln1.")
    if "layer_norm2." in key:
        key = key.replace("layer_norm2.", "ln2.")
    # Map ffn layers
    if "mlp.fc1" in key:
        key = key.replace("mlp.fc1", "linear1")
    if "mlp.fc2" in key:
        key = key.replace("mlp.fc2", "linear2")
    # Fix layernorm typo
    if "pre_layrnorm" in key:
        # Fix typo in weights :)
        key = key.replace("pre_layrnorm", "pre_layernorm")
    if "patch_embedding.weight" in key:
        # Initially, value: [out_channels, in_channels, kH, KW].
        # We want [out_channels, kH, KW, in_channels]
        value = value.permute(0, 2, 3, 1)
    return (key, torch_to_mx(value, dtype=str(value.dtype).replace("torch.", "")))


def should_keep_weight(key: str):
    return not ("position_ids" in key)


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert (OpenAI) CLIP weights to MLX")
    parser.add_argument(
        "--torch-path",
        type=str,
        help="Path to the PyTorch model.",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx_model",
        help="Path to save the MLX model.",
    )

    args = parser.parse_args()

    torch_path = Path(args.torch_path)
    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading")
    torch_weights = torch.load(torch_path / "pytorch_model.bin")
    print("[INFO] Converting")
    mlx_weights = dict([map_weights(k, v) for (k, v) in torch_weights.items()])
    mlx_weights = {k: v for (k, v) in mlx_weights.items() if should_keep_weight(k)}
    print("[INFO] Saving")
    mx.savez(str(mlx_path / "weights.npz"), **mlx_weights)
    shutil.copyfile(
        str(torch_path / "config.json"),
        str(mlx_path / "config.json"),
    )
    shutil.copyfile(
        str(torch_path / "merges.txt"),
        str(mlx_path / "merges.txt"),
    )
    shutil.copyfile(
        str(torch_path / "vocab.json"),
        str(mlx_path / "vocab.json"),
    )
