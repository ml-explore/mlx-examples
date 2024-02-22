# Copyright © 2023-2024 Apple Inc.

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Union

import mlx.core as mx
import torch
from huggingface_hub import snapshot_download


def make_shards(weights: dict, max_file_size_gb: int = 5) -> list:
    max_file_size_bytes = max_file_size_gb << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


def save_weights(save_path: Union[str, Path], weights: Dict[str, Any]) -> None:
    """Save model weights into specified directory."""
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights)
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )

    total_size = sum(v.nbytes for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    for i, shard in enumerate(shards):
        shard_name = shard_file_format.format(i + 1, shards_count)
        shard_path = save_path / shard_name

        mx.save_safetensors(str(shard_path), shard)

        for weight_name in shard.keys():
            index_data["weight_map"][weight_name] = shard_name

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(
            index_data,
            f,
            indent=4,
        )


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
    parser.add_argument(
        "--dtype",
        help="The data type to save the converted model.",
        type=str,
        default="float32",
    )
    args = parser.parse_args()

    torch_path = get_model_path(args.hf_repo)
    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading")
    torch_weights = torch.load(torch_path / "pytorch_model.bin")
    print("[INFO] Converting")
    mlx_weights = {
        k: torch_to_mx(v, dtype=args.dtype) for k, v in torch_weights.items()
    }
    print("[INFO] Saving")
    save_weights(mlx_path, mlx_weights)
    for fn in ["config.json", "merges.txt", "vocab.json", "preprocessor_config.json"]:
        shutil.copyfile(
            str(torch_path / f"{fn}"),
            str(mlx_path / f"{fn}"),
        )
