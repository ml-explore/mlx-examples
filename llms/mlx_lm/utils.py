# Copyright © 2023-2024 Apple Inc.

import copy
import gc
import glob
import importlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer

# Local imports
from .tuner.utils import apply_lora_layers

# Constants
MODEL_REMAPPING = {
    "mistral": "llama",  # mistral is compatible with llama
    "phi-msft": "phixtral",
}

MAX_FILE_SIZE_GB = 5

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
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    try:
        arch = importlib.import_module(f"mlx_lm.models.{model_type}")
    except ImportError:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)

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


def apply_repetition_penalty(logits: mx.array, generated_tokens: Any, penalty: float):
    """
    Apply repetition penalty to specific logits based on the given context.

    Paper: https://arxiv.org/abs/1909.05858

    Args:
        logits (mx.array): The logits produced by the language model.
        generated_tokens (any): A list of N previous tokens.
        penalty (float): The repetition penalty factor to be applied.

    Returns:
        logits (mx.array): Logits with repetition penalty applied to generated tokens.
    """
    if len(generated_tokens) > 0:
        indices = mx.array([token for token in generated_tokens])
        selected_logits = logits[:, indices]
        selected_logits = mx.where(
            selected_logits < 0, selected_logits * penalty, selected_logits / penalty
        )
        logits[:, indices] = selected_logits
    return logits


def generate_step(
    prompt: mx.array,
    model: nn.Module,
    temp: 0.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    top_p: float = 1.0,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling, if 0 the argmax is used.
        repetition_penalty (float, optional): The penalty factor for repeating tokens.
        repetition_context_size (int, optional): The number of tokens to consider for repetition penalty (default 20).

    Yields:
        Generator[Tuple[mx.array, mx.array]]: A generator producing
        one token and probability per call.
    """

    def sample(logits: mx.array) -> Tuple[mx.array, float]:
        softmax_logits = mx.softmax(logits)

        if temp == 0:
            token = mx.argmax(logits, axis=-1)
        else:
            if top_p > 0 and top_p < 1.0:
                if (
                    logits.dtype == mx.bfloat16
                ):  # workdaround for unable to load kernel contiguous_scan_inclusive_sum_bfloat16_bfloat16
                    logits = logits.astype(mx.float32)
                probs = mx.softmax(logits / temp, axis=-1)

                sorted_probs = mx.sort(probs)[::-1]
                sorted_indices = mx.argsort(probs)[::-1]
                cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

                top_probs = mx.where(
                    cumulative_probs > 1 - top_p,
                    sorted_probs,
                    mx.zeros_like(sorted_probs),
                )
                sorted_token = mx.random.categorical(mx.log(top_probs))
                token = sorted_indices.squeeze(0)[sorted_token]
            else:
                token = mx.random.categorical(logits * (1 / temp))

        prob = softmax_logits[0, token]
        return token, prob

    if repetition_penalty and (
        repetition_penalty < 0 or not isinstance(repetition_penalty, float)
    ):
        raise ValueError(
            f"repetition_penalty must be a non-negative float, got {repetition_penalty}"
        )

    y = prompt
    cache = None

    repetition_context = prompt.tolist()

    if repetition_context_size:
        repetition_context = repetition_context[-repetition_context_size:]

    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]

        if repetition_penalty:
            logits = apply_repetition_penalty(
                logits, repetition_context, repetition_penalty
            )
            y, prob = sample(logits)
            repetition_context.append(y.item())
        else:
            y, prob = sample(logits)

        if repetition_context_size:
            if len(repetition_context) > repetition_context_size:
                repetition_context = repetition_context[-repetition_context_size:]
        yield y, prob


def generate(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    temp: float = 0.0,
    max_tokens: int = 100,
    verbose: bool = False,
    formatter: Callable = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = None,
    top_p: float = 1.0,
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
       repetition_penalty (float, optional): The penalty factor for repeating tokens.
       repetition_context_size (int, optional): The number of tokens to consider for repetition penalty.
    """

    if verbose:
        print("=" * 10)
        print("Prompt:", prompt)

    prompt_tokens = mx.array(tokenizer.encode(prompt))

    tic = time.perf_counter()
    tokens = []
    skip = 0
    REPLACEMENT_CHAR = "\ufffd"

    for (token, prob), n in zip(
        generate_step(
            prompt_tokens,
            model,
            temp,
            repetition_penalty,
            repetition_context_size,
            top_p,
        ),
        range(max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break
        if n == 0:
            prompt_time = time.perf_counter() - tic
            tic = time.perf_counter()
        tokens.append(token.item())

        if verbose:
            s = tokenizer.decode(tokens)
            if formatter:
                formatter(s[skip:], prob.item())
                skip = len(s)
            elif REPLACEMENT_CHAR not in s:
                print(s[skip:], end="", flush=True)
                skip = len(s)

    token_count = len(tokens)
    token_string = tokenizer.decode(tokens).replace(REPLACEMENT_CHAR, "")

    if verbose:
        print(token_string[skip:], flush=True)
        gen_time = time.perf_counter() - tic
        print("=" * 10)
        if token_count == 0:
            print("No tokens generated for this prompt")
            return
        prompt_tps = prompt_tokens.size / prompt_time
        gen_tps = (token_count - 1) / gen_time
        print(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
        print(f"Generation: {gen_tps:.3f} tokens-per-sec")

    return token_string


def load_model(model_path: Path, lazy: bool = False) -> nn.Module:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``

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
        # for legacy models that don't have lm_head quant due to non-32 dims
        if "lm_head.scales" not in weights.keys():
            vocab_size = config["vocab_size"]
            extended_linear_class_predicate = (
                lambda layer: linear_class_predicate(layer)
                and layer.weight.shape[0] != vocab_size
            )
            nn.QuantizedLinear.quantize_module(
                model,
                **quantization,
                linear_class_predicate=extended_linear_class_predicate,
            )
        # for models that have lm_head quant
        else:
            nn.QuantizedLinear.quantize_module(
                model,
                **quantization,
                linear_class_predicate=linear_class_predicate,
            )

    model.load_weights(list(weights.items()))

    if not lazy:
        mx.eval(model.parameters())

    model.eval()
    return model


def load(
    path_or_hf_repo: str,
    tokenizer_config={},
    adapter_file: str = None,
    lazy: bool = False,
) -> Tuple[nn.Module, PreTrainedTokenizer]:
    """
    Load the model and tokenizer from a given path or a huggingface repository.

    Args:
        model_path (Path): The path or the huggingface repository to load the model from.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer.
            Defaults to an empty dictionary.
        adapter_file (str, optional): Path to the adapter file. If provided, applies LoRA layers to the model.
            Defaults to None.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
    Returns:
        Tuple[nn.Module, PreTrainedTokenizer]: A tuple containing the loaded model and tokenizer.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    model_path = get_model_path(path_or_hf_repo)

    model = load_model(model_path, lazy)
    if adapter_file is not None:
        model = apply_lora_layers(model, adapter_file)
        model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_config)
    return model, tokenizer


def fetch_from_hub(
    model_path: Path, lazy: bool = False
) -> Tuple[nn.Module, dict, PreTrainedTokenizer]:
    model = load_model(model_path, lazy)

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
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
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


def save_weights(
    save_path: Union[str, Path],
    weights: Dict[str, Any],
    *,
    donate_weights: bool = False,
) -> None:
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

    # Write the weights and make sure no references are kept other than the
    # necessary ones
    if donate_weights:
        weights.clear()
        gc.collect()

    for i in range(len(shards)):
        shard = shards[i]
        shards[i] = None
        shard_name = shard_file_format.format(i + 1, shards_count)
        shard_path = save_path / shard_name

        mx.save_safetensors(str(shard_path), shard)

        for weight_name in shard.keys():
            index_data["weight_map"][weight_name] = shard_name
        del shard
        gc.collect()

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(
            index_data,
            f,
            indent=4,
        )


def quantize_model(
    model: nn.Module, config: dict, q_group_size: int, q_bits: int
) -> Tuple:
    """
    Applies quantization to the model weights.

    Args:
        model (nn.Module): The model to be quantized.
        config (dict): Model configuration.
        q_group_size (int): Group size for quantization.
        q_bits (int): Bits per weight for quantization.

    Returns:
        Tuple: Tuple containing quantized weights and config.
    """
    quantized_config = copy.deepcopy(config)

    nn.QuantizedLinear.quantize_module(
        model, q_group_size, q_bits, linear_class_predicate=linear_class_predicate
    )
    quantized_config["quantization"] = {"group_size": q_group_size, "bits": q_bits}
    quantized_weights = dict(tree_flatten(model.parameters()))

    return quantized_weights, quantized_config


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
