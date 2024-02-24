import glob
import json
import logging
from pathlib import Path
from typing import Generator

import transformers
from huggingface_hub import snapshot_download

import mlx.core as mx
import mlx.nn as nn

from mlx.utils import tree_unflatten

from models.base import BaseModelArgs
from models.tokenizer import Tokenizer, get_tokenizer

import models.whisper as whisper

# Constants
MODEL_MAPPING = {
    "whisper": whisper,
}


def _get_classes(config: dict) -> tuple[nn.Module, BaseModelArgs]:
    """
    Retrieve the model and model args classes based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    model_type = config["model_type"]
    if model_type not in MODEL_MAPPING:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)

    arch = MODEL_MAPPING[model_type]
    return arch.Whisper, arch.ModelDimensions


def load(path_or_hf_repo: str):
    # If the path exists, it will try to load model form it
    # otherwise download and cache from the hf_repo and cache
    model_path = Path(path_or_hf_repo)
    print(f"Loading pretrained model {model_path}")
    if not model_path.exists():
        raise Exception(
            "Could not find model! Did you download model from HF to `mlx_whisper_model` path?"
        )
        # Consider uncommenting the following block if you'd like to dowload the model on-the-fly instead of raising exception
        # model_path = Path(
        #     snapshot_download(
        #         repo_id=path_or_hf_repo,
        #         allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
        #     )
        # )

    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        quantization = config.get("quantization", None)

    weight_files = glob.glob(str(model_path / "*.npz"))
    if len(weight_files) == 0:
        raise FileNotFoundError("No safetensors found in {}".format(model_path))

    weights = {}
    for wf in weight_files:  # todo: whisper-lora: Assert len(weight_files) == 1
        weights.update(mx.load(wf))

    model_class, model_args_class = _get_classes(config=config)
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)
    if quantization is not None:
        nn.QuantizedLinear.quantize_module(
            model,
            **quantization,
            linear_class_predicate=lambda m: isinstance(m, nn.Linear)
            and m.weight.shape[0] != 8,
        )
    weights = tree_unflatten(list(weights.items()))
    model.update(weights)

    mx.eval(model.parameters())
    tokenizer = get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        # language=language, # todo: whisper-lora: pass lang as cli param?
        # task=task, # todo: whisper-lora: what is `task`? Investigate
    )
    return (model, tokenizer, config)


def make_shards(weights: dict, max_file_size_gibibyte: int = 15):
    max_file_size_bytes = max_file_size_gibibyte << 30
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


def save_model(save_dir: str, weights, tokenizer, config):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    shards = make_shards(weights)
    for i, shard in enumerate(shards):
        # TODO use HF file name scheme for simplicity
        mx.savez(str(save_dir / f"weights.npz"), **shard)
        mx.save_safetensors(str(save_dir / f"weights.{i:02d}.safetensors"), shard)
    # todo: whisper-lora: investigate if `tokenizer.save_pretrained(save_dir)` is necessary?
    with open(save_dir / "config.json", "w") as fid:
        json.dump(config, fid, indent=4)
