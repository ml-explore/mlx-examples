# Copyright © 2024 Apple Inc.

import argparse
import json
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from typing import Any, Dict, Union

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

import encodec


def fetch_from_hub(hf_repo: str) -> Path:
    model_path = Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=["*.json", "*.safetensors"],
        )
    )
    return model_path


def upload_to_hub(path: str, upload_repo: str, hf_path: str):
    """
    Uploads the model to Hugging Face hub.

    Args:
        path (str): Local path to the model.
        upload_repo (str): Name of the HF repo to upload to.
        hf_path (str): Path to the original Hugging Face model.
    """
    import os

    from huggingface_hub import HfApi, ModelCard, logging

    content = dedent(
        f"""
        ---
        language: en
        license: other
        library: mlx
        tags:
        - mlx
        ---

        The Model [{upload_repo}](https://huggingface.co/{upload_repo}) was
        converted to MLX format from
        [{hf_path}](https://huggingface.co/{hf_path}).

        This model is intended to be used with the [EnCodec MLX
        example](https://github.com/ml-explore/mlx-examples/tree/main/encodec).
        """
    )

    card = ModelCard(content)
    card.save(os.path.join(path, "README.md"))

    logging.set_verbosity_info()

    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(
        folder_path=path,
        repo_id=upload_repo,
        repo_type="model",
        multi_commits=True,
        multi_commits_verbose=True,
    )
    print(f"Upload successful, go to https://huggingface.co/{upload_repo} for details.")


def save_weights(save_path: Union[str, Path], weights: Dict[str, Any]) -> None:
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    total_size = sum(v.nbytes for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}
    mx.save_safetensors(
        str(save_path / "model.safetensors"), weights, metadata={"format": "mlx"}
    )

    for weight_name in weights.keys():
        index_data["weight_map"][weight_name] = "model.safetensors"

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=4)


def save_config(
    config: dict,
    config_path: Union[str, Path],
) -> None:
    """Save the model configuration to the ``config_path``.

    The final configuration will be sorted before saving for better readability.

    Args:
        config (dict): The model configuration.
        config_path (Union[str, Path]): Model configuration file path.
    """
    # Clean unused keys
    config.pop("_name_or_path", None)

    # sort the config for better readability
    config = dict(sorted(config.items()))

    # write the updated config to the config_path (if provided)
    with open(config_path, "w") as fid:
        json.dump(config, fid, indent=4)


def convert(
    upload: bool,
    model: str,
    dtype: str = None,
):
    hf_repo = f"facebook/encodec_{model}"
    mlx_repo = f"mlx-community/encodec-{model}-{dtype}"
    path = fetch_from_hub(hf_repo)
    save_path = Path("mlx_models")

    weights = mx.load(str(Path(path) / "model.safetensors"))

    with open(path / "config.json", "r") as fid:
        config = SimpleNamespace(**json.load(fid))

    model = encodec.EncodecModel(config)

    new_weights = {}
    for k, v in weights.items():
        basename, pname = k.rsplit(".", 1)
        if pname == "weight_v":
            g = weights[basename + ".weight_g"]
            v = g * (v / mx.linalg.norm(v, axis=(1, 2), keepdims=True))
            k = basename + ".weight"
        elif pname in ["weight_g", "embed_avg", "cluster_size", "inited"]:
            continue
        elif "lstm" in basename:
            w_or_b, ih_or_hh, ln = pname.split("_")
            if w_or_b == "weight":
                new_pname = "Wx" if ih_or_hh == "ih" else "Wh"
            elif w_or_b == "bias" and ih_or_hh == "ih":
                continue
            else:
                v = v + weights[k.replace("_hh_", "_ih_")]
                new_pname = "bias"
            k = basename + "." + ln[1:] + "." + new_pname
        if "conv.weight" in k:
            # Possibly a transposed conv which has a different order
            if "decoder" in k:
                ln = int(k.split(".")[2])
                if "conv" in model.decoder.layers[ln] and isinstance(
                    model.decoder.layers[ln].conv, nn.ConvTranspose1d
                ):
                    v = mx.moveaxis(v, 0, 2)
                else:
                    v = mx.moveaxis(v, 1, 2)
            else:
                v = mx.moveaxis(v, 1, 2)

        new_weights[k] = v
    weights = new_weights

    model.load_weights(list(weights.items()))

    if dtype is not None:
        t = getattr(mx, dtype)
        weights = {k: v.astype(t) for k, v in weights.items()}

    if isinstance(save_path, str):
        save_path = Path(save_path)

    save_weights(save_path, weights)

    save_config(vars(config), config_path=save_path / "config.json")

    if upload:
        upload_to_hub(save_path, mlx_repo, hf_repo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert EnCodec weights to MLX.")
    parser.add_argument(
        "--model",
        type=str,
        default="48khz",
        help="",
        choices=["24khz", "32khz", "48khz"],
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the weights to Hugging Face.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        help="Data type to convert the model to.",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
    )
    args = parser.parse_args()
    convert(upload=args.upload, model=args.model, dtype=args.dtype)
