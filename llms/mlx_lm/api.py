# Copyright © 2023-2024 Apple Inc.

import copy
import glob
import json
import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from .utils import (
    fetch_from_hub,
    get_model_path,
    quantize_model,
    save_weights,
    upload_to_hub,
)


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
    dtype: str = "float16",
    upload_repo: str = None,
):
    print("[INFO] Loading")
    model_path = get_model_path(hf_path)
    model, config, tokenizer = fetch_from_hub(model_path, lazy=True)

    weights = dict(tree_flatten(model.parameters()))
    dtype = mx.float16 if quantize else getattr(mx, dtype)
    weights = {k: v.astype(dtype) for k, v in weights.items()}

    if quantize:
        print("[INFO] Quantizing")
        model.load_weights(list(weights.items()))
        weights, config = quantize_model(model, config, q_group_size, q_bits)

    if isinstance(mlx_path, str):
        mlx_path = Path(mlx_path)

    del model
    save_weights(mlx_path, weights, donate_weights=True)

    py_files = glob.glob(str(model_path / "*.py"))
    for file in py_files:
        shutil.copy(file, mlx_path)

    tokenizer.save_pretrained(mlx_path)

    with open(mlx_path / "config.json", "w") as fid:
        json.dump(config, fid, indent=4)

    if upload_repo is not None:
        upload_to_hub(mlx_path, upload_repo, hf_path)
