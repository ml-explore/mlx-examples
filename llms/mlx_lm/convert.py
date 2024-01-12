import argparse
import copy
import glob
import json
from pathlib import Path
from typing import Dict, Tuple

import mlx.core as mx
import mlx.nn as nn
import transformers
from mlx.utils import tree_flatten

from .utils import get_model_path, load

MAX_FILE_SIZE_GB = 15


def configure_parser() -> argparse.ArgumentParser:
    """
    Configures and returns the argument parser for the script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face model to MLX format"
    )

    parser.add_argument("--hf-path", type=str, help="Path to the Hugging Face model.")
    parser.add_argument(
        "--mlx-path", type=str, default="mlx_model", help="Path to save the MLX model."
    )
    parser.add_argument(
        "-q", "--quantize", help="Generate a quantized model.", action="store_true"
    )
    parser.add_argument(
        "--q-group-size", help="Group size for quantization.", type=int, default=64
    )
    parser.add_argument(
        "--q-bits", help="Bits per weight for quantization.", type=int, default=4
    )
    parser.add_argument(
        "--dtype",
        help="Type to save the parameters, ignored if -q is given.",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )
    return parser


def fetch_from_hub(
    model_path: str,
) -> Tuple[Dict, dict, transformers.PreTrainedTokenizer]:
    model_path = get_model_path(model_path)

    weight_files = glob.glob(f"{model_path}/*.safetensors")
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf).items())

    config = transformers.AutoConfig.from_pretrained(model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    return weights, config.to_dict(), tokenizer


def quantize_model(
    weights: dict, config: dict, hf_path: str, q_group_size: int, q_bits: int
) -> tuple:
    """
    Applies quantization to the model weights.

    Args:
        weights (dict): Model weights.
        config (dict): Model configuration.
        hf_path (str): HF model path..
        q_group_size (int): Group size for quantization.
        q_bits (int): Bits per weight for quantization.

    Returns:
        tuple: Tuple containing quantized weights and config.
    """
    quantized_config = copy.deepcopy(config)
    model, _ = load(hf_path)
    model.load_weights(list(weights.items()))

    nn.QuantizedLinear.quantize_module(model, q_group_size, q_bits)
    quantized_config["quantization"] = {
        "group_size": q_group_size,
        "bits": q_bits,
    }
    quantized_weights = dict(tree_flatten(model.parameters()))

    return quantized_weights, quantized_config


def make_shards(weights: dict, max_file_size_gb: int = MAX_FILE_SIZE_GB) -> list:
    """
    Splits the weights into smaller shards.

    Args:
        weights (dict): Model weights.
        max_file_size_gb (int): Maximum size of each shard in gigabytes.

    Returns:
        list: List of weight shards.
    """
    max_file_size_bytes = max_file_size_gb << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        estimated_size = v.size * v.dtype.size
        if shard_size + estimated_size > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += estimated_size
    shards.append(shard)
    return shards


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

    card = ModelCard.load(hf_path)
    card.data.tags = ["mlx"] if card.data.tags is None else card.data.tags + ["mlx"]
    card.text = f"""
# {upload_repo}
This model was converted to MLX format from [`{hf_path}`]().
Refer to the [original model card](https://huggingface.co/{hf_path}) for more details on the model.
## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("{upload_repo}")
response = generate(model, tokenizer, prompt="hello", verbose=True)
```
"""
    card.save(os.path.join(path, "README.md"))

    logging.set_verbosity_info()

    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(
        folder_path=path,
        repo_id=upload_repo,
        repo_type="model",
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
    weights, config, tokenizer = fetch_from_hub(hf_path)
    dtype = mx.float16 if quantize else getattr(mx, dtype)
    weights = {k: v.astype(dtype) for k, v in weights.items()}
    if quantize:
        print("[INFO] Quantizing")
        weights, config = quantize_model(weights, config, hf_path, q_group_size, q_bits)

    mlx_path = Path(mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)
    shards = make_shards(weights)
    for i, shard in enumerate(shards):
        mx.save_safetensors(str(mlx_path / f"weights.{i:02d}.safetensors"), shard)
    tokenizer.save_pretrained(mlx_path)
    with open(mlx_path / "config.json", "w") as fid:
        json.dump(config, fid, indent=4)

    if upload_repo is not None:
        upload_to_hub(mlx_path, upload_repo, hf_path)


if __name__ == "__main__":
    parser = configure_parser()
    args = parser.parse_args()
    convert(**vars(args))
