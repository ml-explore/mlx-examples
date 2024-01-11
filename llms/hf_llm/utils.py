import glob
import json
import logging
from pathlib import Path
from typing import Generator, Tuple

import mlx.core as mx
import mlx.nn as nn

# Local imports
import models.llama as llama
import models.phi2 as phi2
from huggingface_hub import snapshot_download
from models.base import BaseModelArgs
from transformers import AutoTokenizer, PreTrainedTokenizer

# Constants
MODEL_MAPPING = {
    "llama": llama,
    "mistral": llama,  # mistral is compatible with llama
    "phi": phi2,
}


def _get_classes(config: dict):
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
    return arch.Model, arch.ModelArgs


def get_model_path(path_or_hf_repo: str) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.

    Returns:
        Path: The path to the model.
    """
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                allow_patterns=["*.json", "*.safetensors", "*.py", "tokenizer.model"],
            )
        )
    return model_path


def generate(
    prompt: mx.array, model: nn.Module, temp: float = 0.0
) -> Generator[mx.array, None, None]:
    """
    Generate text based on the given prompt and model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling. If temp is 0, use max sampling.

    Yields:
        mx.array: The generated text.
    """

    def sample(logits: mx.array) -> mx.array:
        return (
            mx.argmax(logits, axis=-1)
            if temp == 0
            else mx.random.categorical(logits * (1 / temp))
        )

    y = prompt
    cache = None
    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]
        y = sample(logits)
        yield y


def load(path_or_hf_repo: str) -> Tuple[nn.Module, PreTrainedTokenizer]:
    """
    Load the model from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (str): The path or the huggingface repository to load the model from.

    Returns:
        Tuple[nn.Module, PreTrainedTokenizer]: The loaded model and tokenizer.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    model_path = get_model_path(path_or_hf_repo)

    try:
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
            quantization = config.get("quantization", None)
    except FileNotFoundError:
        logging.error(f"Config file not found in {model_path}")
        raise
    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if not weight_files:
        logging.error(f"No safetensors found in {model_path}")
        raise FileNotFoundError(f"No safetensors found in {model_path}")
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    model_class, model_args_class = _get_classes(config=config)

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    if quantization is not None:
        nn.QuantizedLinear.quantize_module(model, **quantization)

    model.load_weights(list(weights.items()))

    mx.eval(model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer
