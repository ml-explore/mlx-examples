import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Union

import mlx.core as mx
from huggingface_hub import snapshot_download


def save_weights(save_path: Union[str, Path], weights: Dict[str, mx.array]) -> None:
    """Save model weights into specified directory."""
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    total_size = sum(v.nbytes for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    model_path = save_path / "model.safetensors"
    mx.save_safetensors(str(model_path), weights)

    for weight_name in weights.keys():
        index_data["weight_map"][weight_name] = "model.safetensors"

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=4)


def download(hf_repo):
    return Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=["*.safetensors", "*.json"],
            resume_download=True,
        )
    )


def convert(model_path):
    weight_file = str(model_path / "model.safetensors")
    weights = mx.load(weight_file)

    mlx_weights = dict()
    for k, v in weights.items():
        if k in {
            "vision_encoder.patch_embed.projection.weight",
            "vision_encoder.neck.conv1.weight",
            "vision_encoder.neck.conv2.weight",
            "prompt_encoder.mask_embed.conv1.weight",
            "prompt_encoder.mask_embed.conv2.weight",
            "prompt_encoder.mask_embed.conv3.weight",
        }:
            v = v.transpose(0, 2, 3, 1)
        if k in {
            "mask_decoder.upscale_conv1.weight",
            "mask_decoder.upscale_conv2.weight",
        }:
            v = v.transpose(1, 2, 3, 0)
        mlx_weights[k] = v
    return mlx_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Meta SAM weights to MLX")
    parser.add_argument(
        "--hf-path",
        default="facebook/sam-vit-base",
        type=str,
        help="Path to the Hugging Face model repo.",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="sam-vit-base",
        help="Path to save the MLX model.",
    )
    args = parser.parse_args()

    model_path = download(args.hf_path)

    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    mlx_weights = convert(model_path)
    save_weights(mlx_path, mlx_weights)
    shutil.copy(model_path / "config.json", mlx_path / "config.json")
