# Copyright © 2023-2024 Apple Inc.

import argparse
import shutil
from pathlib import Path
from typing import Tuple

import mlx.core as mx
import torch
from huggingface_hub import snapshot_download
from mlx_lm.utils import save_weights


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
        help="The data type to save the converted model. "
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
