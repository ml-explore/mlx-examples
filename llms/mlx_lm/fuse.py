import argparse
import glob
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Union

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .tuner.linear import LoRALinear
from .tuner.utils import apply_lora_layers
from .utils import (
    fetch_from_hub,
    get_model_path,
    make_shards,
    save_weights,
    upload_to_hub,
)


def save_model(
    base_model_path: Union[str, Path],
    save_path: str,
    weights: Dict[str, Any],
    tokenizer: Any,
    config: Dict[str, Any],
) -> None:
    if isinstance(base_model_path, str):
        base_model_path = Path(base_model_path)

    if isinstance(save_path, str):
        save_path = Path(save_path)

    save_weights(save_path, weights)

    py_files = glob.glob(str(base_model_path / "*.py"))
    for file in py_files:
        shutil.copy(file, save_path)

    tokenizer.save_pretrained(save_path)
    with open(save_path / "config.json", "w") as fid:
        json.dump(config, fid, indent=4)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--save-path",
        default="lora_fused_model",
        help="The path to save the fused model.",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        default="adapters.npz",
        help="Path to the trained adapter weights (npz or safetensors).",
    )
    parser.add_argument(
        "--hf-path",
        type=str,
        default=None,
        help="Path to the original Hugging Face model. Required for upload if --model is a local directory.",
    )
    parser.add_argument(
        "--upload-name",
        type=str,
        default=None,
        help="The name of model to upload to Hugging Face MLX Community",
    )
    return parser.parse_args()


def main() -> None:
    print("Loading pretrained model")
    args = parse_arguments()

    model_path = get_model_path(args.model)
    model, config, tokenizer = fetch_from_hub(model_path)

    model.freeze()
    model = apply_lora_layers(model, args.adapter_file)
    fused_linears = [
        (n, m.to_linear())
        for n, m in model.named_modules()
        if isinstance(m, LoRALinear)
    ]

    model.update_modules(tree_unflatten(fused_linears))
    weights = dict(tree_flatten(model.parameters()))
    save_model(
        base_model_path=model_path,
        save_path=args.save_path,
        weights=weights,
        tokenizer=tokenizer,
        config=config,
    )

    if args.upload_name is not None:
        hf_path = args.hf_path or (
            args.model if not Path(args.model).exists() else None
        )
        if hf_path is None:
            raise ValueError(
                "Must provide original Hugging Face repo to upload local model."
            )
        upload_to_hub(args.save_path, args.upload_name, hf_path)


if __name__ == "__main__":
    main()
