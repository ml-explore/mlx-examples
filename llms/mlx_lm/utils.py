import copy
import glob
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer

# Local imports
from .models import llama, mixtral, phi2, plamo, qwen
from .tuner.utils import apply_lora_layers

# Constants
MODEL_MAPPING = {
    "llama": llama,
    "mistral": llama,  # mistral is compatible with llama
    "mixtral": mixtral,
    "phi": phi2,
    "qwen": qwen,
    "plamo": plamo,
}
MAX_FILE_SIZE_GB = 15

linear_class_predicate = (
    lambda m: isinstance(m, nn.Linear)
    and m.weight.shape[0]
    != 8  # avoid quantizing gate layers, otherwise we have to re-quant and upload all the mixtral models
)


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
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.py",
                    "tokenizer.model",
                    "*.tiktoken",
                ],
            )
        )
    return model_path


def generate_step(
    prompt: mx.array,
    model: nn.Module,
    temp: float = 0.0,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling, if 0 the argmax is used.
    Yields:
        Generator[Tuple[mx.array, mx.array]]: A generator producing
        one token and probability per call.
    """

    def sample(logits: mx.array) -> Tuple[mx.array, float]:
        softmax_logits = mx.softmax(logits)

        if temp == 0:
            token = mx.argmax(logits, axis=-1)
        else:
            token = mx.random.categorical(logits * (1 / temp))

        prob = softmax_logits[0, token]
        return token, prob

    y = prompt
    cache = None
    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]
        y, prob = sample(logits)
        yield y, prob


def generate(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    temp: float = 0.0,
    max_tokens: int = 100,
    verbose: bool = False,
    formatter: Callable = None,
) -> str:
    """
    Generate text from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       temp (float): The temperature for sampling (default 0).
       max_tokens (int): The maximum number of tokens (default 100).
       verbose (bool): If ``True``, print tokens and timing information
           (default ``False``).
       formatter (Optional[Callable]): A function which takes a token and a
           probability and displays it.
    """

    if verbose:
        print("=" * 10)
        print("Prompt:", prompt)

    prompt = mx.array(tokenizer.encode(prompt))

    tic = time.time()
    tokens = []
    skip = 0
    REPLACEMENT_CHAR = "\ufffd"

    for (token, prob), n in zip(generate_step(prompt, model, temp), range(max_tokens)):
        if token == tokenizer.eos_token_id:
            break
        if n == 0:
            prompt_time = time.time() - tic
            tic = time.time()
        tokens.append(token.item())

        if verbose:
            s = tokenizer.decode(tokens)
            if formatter:
                formatter(s[skip:], prob.item())
                skip = len(s)
            elif REPLACEMENT_CHAR not in s:
                print(s[skip:], end="", flush=True)
                skip = len(s)

    tokens = tokenizer.decode(tokens).replace(REPLACEMENT_CHAR, "")

    if verbose:
        print(tokens[skip:], flush=True)
        gen_time = time.time() - tic
        print("=" * 10)
        if len(tokens) == 0:
            print("No tokens generated for this prompt")
            return
        prompt_tps = prompt.size / prompt_time
        gen_tps = (len(tokens) - 1) / gen_time
        print(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
        print(f"Generation: {gen_tps:.3f} tokens-per-sec")

    return tokens


def load_model(model_path: Path) -> nn.Module:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.

    Returns:
        nn.Module: The loaded and initialized model.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """
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
    if hasattr(model_class, "sanitize"):
        weights = model_class.sanitize(weights)

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    if quantization is not None:
        nn.QuantizedLinear.quantize_module(
            model,
            **quantization,
            linear_class_predicate=linear_class_predicate,
        )

    model.load_weights(list(weights.items()))

    mx.eval(model.parameters())

    model.eval()
    return model


def load(
    path_or_hf_repo: str, tokenizer_config={}, adapter_file: str = None
) -> Tuple[nn.Module, PreTrainedTokenizer]:
    """
    Load the model and tokenizer from a given path or a huggingface repository.

    Args:
        model_path (Path): The path or the huggingface repository to load the model from.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer.
            Defaults to an empty dictionary.
        adapter_file (str, optional): Path to the adapter file. If provided, applies LoRA layers to the model.
            Defaults to None.
    Returns:
        Tuple[nn.Module, PreTrainedTokenizer]: A tuple containing the loaded model and tokenizer.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    model_path = get_model_path(path_or_hf_repo)

    model = load_model(model_path)
    if adapter_file is not None:
        model = apply_lora_layers(model, adapter_file)

    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_config)
    return model, tokenizer


def fetch_from_hub(
    model_path: Path,
) -> Tuple[Dict, dict, PreTrainedTokenizer]:
    model = load_model(model_path)

    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, config.to_dict(), tokenizer


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

    for i, shard in enumerate(shards):
        shard_name = shard_file_format.format(i + 1, shards_count)
        mx.save_safetensors(str(save_path / shard_name), shard)
