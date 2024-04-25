# Copyright © 2023-2024 Apple Inc.

import argparse
import glob
import shutil
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import yaml
from mlx.utils import tree_flatten, tree_map

from .utils import (
    fetch_from_hub,
    get_model_path,
    save_config,
    save_weights,
    upload_to_hub,
)


def configure_parser() -> argparse.ArgumentParser:
    """
    Configures and returns the argument parser for the script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Merge multiple models.")

    parser.add_argument("--config", type=str, help="Path to the YAML config.")
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx_merged_model",
        help="Path to save the MLX model.",
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )
    return parser


def slerp(t, w1, w2, eps=1e-5):
    """
    Spherical linear interpolation

    Args:
        t (float): Interpolation weight in [0.0, 1.0]
        w1 (mx.array): First input
        w2 (mx.array): Second input
        eps (float): Constant for numerical stability
    Returns:
        mx.array: Interpolated result
    """
    t = float(t)
    if t == 0:
        return w1
    elif t == 1:
        return w2
    # Normalize
    v1 = w1 / mx.linalg.norm(w1)
    v2 = w2 / mx.linalg.norm(w2)
    # Angle
    dot = mx.clip((v1 * v2).sum(), 0.0, 1.0)
    theta = mx.arccos(dot)
    sin_theta = mx.sin(theta + eps)
    s1 = mx.sin(theta * (1 - t)) / sin_theta
    s2 = mx.sin(theta * t) / sin_theta
    return s1 * w1 + s2 * w2


def merge_models(base_model: nn.Module, model: nn.Module, config: dict):
    method = config.get("method", None)
    if method != "slerp":
        raise ValueError(f"Merge method {method} not supported")

    num_layers = len(model.layers)

    def unpack_values(vals):
        if isinstance(vals, (int, float)):
            return np.full(num_layers, vals)
        bins = len(vals) - 1
        sizes = [num_layers // bins] * bins
        sizes[-1] = num_layers - sum(sizes[:-1])
        return np.concatenate(
            [np.linspace(v1, v2, s) for v1, v2, s in zip(vals[:-1], vals[1:], sizes)]
        )

    param_list = config["parameters"]["t"]
    params = {}
    filter_keys = set()
    for pl in param_list[:-1]:
        params[pl["filter"]] = unpack_values(pl["value"])
        filter_keys.add(pl["filter"])
    default = unpack_values(param_list[-1]["value"])

    for e in range(num_layers):
        bl = base_model.layers[e]
        l = model.layers[e]
        base_weights = bl.parameters()
        weights = l.parameters()
        for k, w1 in base_weights.items():
            w2 = weights[k]
            t = params.get(k, default)[e]
            base_weights[k] = tree_map(lambda x, y: slerp(t, x, y), w1, w2)
        base_model.update(base_weights)


def merge(
    config: str,
    mlx_path: str = "mlx_model",
    upload_repo: Optional[str] = None,
):
    with open(config, "r") as fid:
        merge_conf = yaml.safe_load(fid)
    print("[INFO] Loading")

    model_paths = merge_conf.get("models", [])
    if len(model_paths) < 2:
        raise ValueError(f"Expected at least 2 models, got {len(model_paths)}.")

    # Load all models
    base_hf_path = model_paths[0]
    base_path = get_model_path(base_hf_path)
    base_model, base_config, tokenizer = fetch_from_hub(base_path, lazy=True)
    models = []
    for mp in model_paths[1:]:
        model, model_config, _ = fetch_from_hub(get_model_path(mp), lazy=True)
        base_type = base_config["model_type"]
        model_type = model_config["model_type"]
        if base_type != model_type:
            raise ValueError(
                f"Can only merge models of the same type,"
                f" but got {base_type} and {model_type}."
            )
        models.append(model)

    # Merge models into base model
    for m in models:
        merge_models(base_model, m, merge_conf)

    # Save base model
    mlx_path = Path(mlx_path)
    weights = dict(tree_flatten(base_model.parameters()))
    del models, base_model
    save_weights(mlx_path, weights, donate_weights=True)
    py_files = glob.glob(str(base_path / "*.py"))
    for file in py_files:
        shutil.copy(file, mlx_path)

    tokenizer.save_pretrained(mlx_path)

    save_config(config, config_path=mlx_path / "config.json")

    if upload_repo is not None:
        upload_to_hub(mlx_path, upload_repo, base_hf_path)


def main():
    parser = configure_parser()
    args = parser.parse_args()
    merge(**vars(args))


if __name__ == "__main__":
    main()
