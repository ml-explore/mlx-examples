# Copyright © 2023-2024 Apple Inc.

import argparse
import shutil
from pathlib import Path
from typing import Tuple

import mlx.core as mx
import torch
from huggingface_hub import snapshot_download


def get_model_path(path_or_hf_repo: str) -> Path:
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                allow_patterns=[
                    "*.bin",
                    "*.json",
                    "*.txt",
                ],
            )
        )
    return model_path


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
    parser = argparse.ArgumentParser(
        description="Download and Convert (OpenAI) CLIP weights to MLX"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Hugging Face repository name.",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx_model",
        help="Path to save the MLX model.",
    )

    args = parser.parse_args()

    torch_path = get_model_path(args.hf_repo)
    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading")
    torch_weights = torch.load(torch_path / "pytorch_model.bin")
    print("[INFO] Converting")
    mlx_weights = dict(map_weights(k, v) for (k, v) in torch_weights.items())
    mlx_weights = {k: v for (k, v) in mlx_weights.items() if should_keep_weight(k)}
    print("[INFO] Saving")
    mx.savez(str(mlx_path / "weights.npz"), **mlx_weights)
    for fn in ["config.json", "merges.txt", "vocab.json", "preprocessor_config.json"]:
        shutil.copyfile(
            str(torch_path / f"{fn}"),
            str(mlx_path / f"{fn}"),
        )
