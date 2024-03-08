import argparse
import json
from pathlib import Path
from typing import Any, Dict, Union

import mlx.core as mx
import torch


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


def pt2mlx(torch_weights):
    mlx_weights = dict()
    for k, v in torch_weights.items():
        k = k.replace("image_encoder.neck.", "image_encoder.neck.layers.")
        k = k.replace(
            "mask_decoder.output_upscaling.", "mask_decoder.output_upscaling.layers."
        )
        k = k.replace(
            "prompt_encoder.mask_downscaling.",
            "prompt_encoder.mask_downscaling.layers.",
        )
        v = mx.array(v.numpy())
        if k in {
            "image_encoder.patch_embed.proj.weight",
            "image_encoder.neck.layers.0.weight",
            "image_encoder.neck.layers.2.weight",
            "prompt_encoder.mask_downscaling.layers.0.weight",
            "prompt_encoder.mask_downscaling.layers.3.weight",
            "prompt_encoder.mask_downscaling.layers.6.weight",
        }:
            v = v.transpose(0, 2, 3, 1)
        if k in {
            "mask_decoder.output_upscaling.layers.0.weight",
            "mask_decoder.output_upscaling.layers.3.weight"
        }:
            v = v.transpose(1, 2, 3, 0)
        mlx_weights[k] = v
    return mlx_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Meta SAM weights to MLX")
    parser.add_argument(
        "--pt-path",
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

    torch_path = args.pt_path
    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    print("Loading")
    torch_weights = torch.load(torch_path)
    print("Converting")
    mlx_weights = pt2mlx(torch_weights)
    print("Saving")
    save_weights(mlx_path, mlx_weights)
