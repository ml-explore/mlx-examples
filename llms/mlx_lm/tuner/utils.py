# Copyright © 2024 Apple Inc.
import json
import types
from pathlib import Path
from typing import Dict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_unflatten

from .lora import LoRALinear


def build_schedule(schedule_config: Dict):
    """
    Build a learning rate schedule from the given config.
    """
    schedule_fn = getattr(opt.schedulers, schedule_config["name"])
    arguments = schedule_config["arguments"]
    initial_lr = arguments[0]
    bound_schedule_fn = schedule_fn(*arguments)
    if warmup_steps := schedule_config.get("warmup", 0):
        warmup_init = schedule_config.get("warmup_init", 0.0)
        warmup_fn = opt.schedulers.linear_schedule(
            warmup_init, initial_lr, warmup_steps
        )
        return opt.schedulers.join_schedules(
            [warmup_fn, bound_schedule_fn], [warmup_steps + 1]
        )
    else:
        return bound_schedule_fn


def linear_to_lora_layers(
    model: nn.Module,
    num_lora_layers: int,
    config: Dict,
):
    """
    Convert some of the models linear layers to lora layers.

    Args:
        model (nn.Module): The neural network model.
        num_lora_layers (int): The number of blocks to convert to lora layers
        starting from the last layer.
        config (dict): More configuration parameters for LoRA, including the
          rank, alpha, scale, and optional layer keys.
    """

    num_layers = len(model.layers)
    if num_lora_layers > num_layers:
        raise ValueError(
            f"Requested {num_lora_layers} LoRA layers "
            f"but the model only has {num_layers} layers."
        )

    to_lora = lambda lin: LoRALinear.from_linear(
        lin,
        r=config["rank"],
        alpha=config["alpha"],
        scale=config["scale"],
        dropout=config["dropout"],
    )

    keys = config.get("keys", None)
    if keys is not None:
        keys = set(keys)
    elif model.model_type in [
        "mistral",
        "llama",
        "phi",
        "mixtral",
        "stablelm",
        "qwen2",
        "qwen2_moe",
        "gemma",
        "starcoder2",
        "cohere",
        "minicpm",
    ]:
        keys = set(["self_attn.q_proj", "self_attn.v_proj"])
        if model.model_type == "mixtral":
            keys.add("block_sparse_moe.gate")
        if model.model_type == "qwen2_moe":
            keys.add("mlp.gate")
            keys.add("mlp.shared_expert_gate")
    elif model.model_type == "olmo":
        keys = set(["att_proj"])
    elif model.model_type in ["phi3", "openelm"]:
        keys = set(["self_attn.qkv_proj"])
    elif model.model_type == "phi-msft":
        keys = set(["mixer.Wqkv", "moe.gate"])
    elif model.model_type == "dbrx":
        keys = set(["norm_attn_norm.attn.Wqkv", "ffn.router.layer"])
    else:
        raise ValueError(f"Lora does not support {model.model_type}")

    for l in model.layers[num_layers - num_lora_layers :]:
        lora_layers = [(k, to_lora(m)) for k, m in l.named_modules() if k in keys]
        l.update_modules(tree_unflatten(lora_layers))


def apply_lora_layers(model: nn.Module, adapter_path: str) -> nn.Module:
    """
    Apply LoRA layers to the model.

    Args:
        model (nn.Module): The neural network model.
        adapter_path (str): Path to the adapter configuration file.

    Returns:
        nn.Module: The updated model with LoRA layers applied.
    """
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"The adapter path does not exist: {adapter_path}")
    with open(adapter_path / "adapter_config.json", "r") as fid:
        config = types.SimpleNamespace(**json.load(fid))
    linear_to_lora_layers(model, config.lora_layers, config.lora_parameters)
    model.load_weights(str(adapter_path / "adapters.safetensors"), strict=False)
    return model


def dequantize(model: nn.Module) -> nn.Module:
    """
    Dequantize the quantized linear layers in the model.

    Args:
        model (nn.Module): The model with quantized linear layers.

    Returns:
        nn.Module: The model with dequantized layers.
    """
    de_quantize_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            bias = "bias" in module
            weight = module.weight
            weight = mx.dequantize(
                weight,
                module.scales,
                module.biases,
                module.group_size,
                module.bits,
            ).astype(mx.float16)
            output_dims, input_dims = weight.shape
            linear = nn.Linear(input_dims, output_dims, bias=bias)
            linear.weight = weight
            if bias:
                linear.bias = module.bias
            de_quantize_layers.append((name, linear))
        if isinstance(module, nn.QuantizedEmbedding):
            weight = mx.dequantize(
                module.weight,
                module.scales,
                module.biases,
                module.group_size,
                module.bits,
            ).astype(mx.float16)
            num_embeddings, dims = weight.shape
            emb = nn.Embedding(num_embeddings, dims)
            emb.weight = weight
            de_quantize_layers.append((name, emb))

    if len(de_quantize_layers) > 0:
        model.update_modules(tree_unflatten(de_quantize_layers))
    return model


def remove_lora_layers(model: nn.Module) -> nn.Module:
    """
    Remove the LoRA layers from the model.

    Args:
        model (nn.Module): The model with LoRA layers.

    Returns:
        nn.Module: The model without LoRA layers.
    """
    reset_layers = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            reset_layers.append((name, module.linear))
    if len(reset_layers) > 0:
        model.update_modules(tree_unflatten(reset_layers))
    return model
