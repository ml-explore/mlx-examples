import argparse
import subprocess
from pathlib import Path
from typing import Tuple

import mlx.core as mx
from mixer import MLX_WEIGHTS_PATH, MODELS


def download_weights_if_necessary(model: str, weights_path: Path) -> Path:
    model_weights_path = weights_path / f"{model}.npz"

    if not model_weights_path.exists():
        weights_path.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gsutil", "cp", MODELS[model]["weights"], model_weights_path])

    return model_weights_path


def map_weights(key: str, value: mx.array) -> Tuple[str, mx.array]:
    def rename_stem(weight: str) -> str:
        weight = weight.replace("stem", "patch_embedding.conv")
        return weight

    def rename_linear(weight: str) -> str:
        weight = weight.replace("Dense_0", "layers.0")
        weight = weight.replace("Dense_1", "layers.2")
        weight = weight.replace("kernel", "weight")
        return weight

    def rename_layer_norm(weight: str) -> str:
        weight = weight.replace("LayerNorm_", "ln")
        weight = weight.replace("layer_norm", "ln")
        weight = weight.replace("scale", "weight")
        return weight

    def rename_mixer_block(weight: str) -> str:
        weight = weight.replace("MixerBlock_", "blocks.")
        return weight

    def rename_weight(weight: str) -> str:
        weight = weight.replace("/", ".")
        weight = rename_stem(weight)
        weight = rename_mixer_block(weight)
        weight = rename_linear(weight)
        weight = rename_layer_norm(weight)
        return weight

    key = rename_weight(key)

    if "conv.weight" in key:
        # Initially, value: [kH, KW, in_channels, out_channels].
        # We want [out_channels, kH, KW, in_channels]
        value = value.transpose(3, 0, 1, 2)
    elif "weight" in key:
        if value.ndim == 2:
            # Initially, value: [in_channels, out_channels].
            # We want [out_channels, in_channels]
            value = value.transpose(1, 0)

    return (key, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and Convert (Google) MLPMixer weights to MLX"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="imagenet1k-MixerB-16",
        choices=MODELS.keys(),
        help="Checkpoint.",
    )
    parser.add_argument(
        "--jax_path",
        type=str,
        default="weights/jax",
        help="Path to download the original Jax model. Default: weights/jax.",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default=MLX_WEIGHTS_PATH,
        help="Path to save the MLX model. Default: weights/mlx.",
    )

    args = parser.parse_args()

    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading")
    original_weights_path = download_weights_if_necessary(
        args.model, Path(args.jax_path)
    )
    jax_weights = mx.load(str(original_weights_path))
    print("[INFO] Converting")
    mlx_weights = dict(map_weights(k, v) for (k, v) in jax_weights.items())
    print("[INFO] Saving")
    model_mlx_weights_path = str(mlx_path / f"{args.model}.npz")
    mx.savez(model_mlx_weights_path, **mlx_weights)
