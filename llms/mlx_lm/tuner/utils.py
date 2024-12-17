# Copyright © 2024 Apple Inc.
import json
import types
from pathlib import Path
from typing import Dict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

from ..models.switch_layers import QuantizedSwitchLinear, SwitchLinear
from .dora import DoRAEmbedding, DoRALinear
from .lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear


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
    num_layers: int,
    config: Dict,
    use_dora: bool = False,
):
    """
    Convert some of the models linear layers to lora layers.

    Args:
        model (nn.Module): The neural network model.
        num_layers (int): The number of blocks to convert to lora layers
        starting from the last layer.
        config (dict): More configuration parameters for LoRA, including the
          rank, scale, and optional layer keys.
        use_dora (bool): If True, uses DoRA instead of LoRA.
          Default: ``False``
    """
    if num_layers > len(model.layers):
        raise ValueError(
            f"Requested {num_layers} LoRA layers "
            f"but the model only has {len(model.layers)} layers."
        )

    def to_lora(layer):
        if isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
            LoRALayer = DoRALinear if use_dora else LoRALinear
        elif isinstance(layer, (SwitchLinear, QuantizedSwitchLinear)):
            if use_dora:
                raise ValueError(f"{type(layer).__name__} doesn't support DoRA yet.")
            LoRALayer = LoRASwitchLinear
        elif isinstance(layer, (nn.Embedding, nn.QuantizedEmbedding)):
            LoRALayer = DoRAEmbedding if use_dora else LoRAEmbedding
        else:
            raise ValueError(
                f"Can't convert layer of type {type(layer).__name__} to LoRA"
            )

        return LoRALayer.from_base(
            layer,
            r=config["rank"],
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
        "nemotron",
        "stablelm",
        "qwen2",
        "qwen2_moe",
        "phimoe",
        "gemma",
        "gemma2",
        "starcoder2",
        "cohere",
        "cohere2",
        "minicpm",
        "deepseek",
        "olmo2",
    ]:
        keys = set(["self_attn.q_proj", "self_attn.v_proj"])
        if model.model_type in ["mixtral", "phimoe"]:
            keys.add("block_sparse_moe.gate")
        if model.model_type == "qwen2_moe":
            keys.add("mlp.gate")
            keys.add("mlp.shared_expert_gate")

    elif model.model_type == "gpt_bigcode":
        keys = set(["attn.c_attn"])
    elif model.model_type == "gpt2":
        keys = set(["attn.c_attn"])
    elif model.model_type == "gpt_neox":
        keys = set(["attention.query_key_value"])
    elif model.model_type == "olmo":
        keys = set(["att_proj"])
    elif model.model_type == "openelm":
        keys = set(["attn.qkv_proj"])
    elif model.model_type == "phi3":
        keys = set(["self_attn.qkv_proj"])
    elif model.model_type == "phi-msft":
        keys = set(["mixer.Wqkv", "moe.gate"])
    elif model.model_type == "dbrx":
        keys = set(["norm_attn_norm.attn.Wqkv", "ffn.router.layer"])
    elif model.model_type == "internlm2":
        keys = set(["attention.wqkv", "attention.wo"])
    elif model.model_type == "deepseek_v2":
        keys = set(
            [
                "self_attn.q_proj",
                "self_attn.q_a_proj",
                "self_attn.q_b_proj",
                "self_attn.kv_a_proj_with_mqa",
                "self_attn.kv_b_proj",
            ]
        )
    elif model.model_type == "mamba":
        keys = set(
            [
                "mixer.in_proj",
                "mixer.x_proj",
                "mixer.dt_proj",
                "mixer.out_proj",
            ]
        )
    elif model.model_type == "mamba2":
        keys = set(
            [
                "mixer.in_proj",
                "mixer.out_proj",
            ]
        )
    elif model.model_type == "exaone":
        keys = set(["attn.attention.q_proj", "attn.attention.v_proj"])
    else:
        raise ValueError(f"Lora does not support {model.model_type}")

    for l in model.layers[-min(num_layers, 0) :]:
        lora_layers = [(k, to_lora(m)) for k, m in l.named_modules() if k in keys]
        if lora_layers:
            l.update_modules(tree_unflatten(lora_layers))

    lora_modules = [(k, to_lora(m)) for k, m in model.named_modules() if k in keys]
    if lora_modules:
        model.update_modules(tree_unflatten(lora_modules))


def load_adapters(model: nn.Module, adapter_path: str) -> nn.Module:
    """
    Load any fine-tuned adapters / layers.

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
    fine_tune_type = getattr(config, "fine_tune_type", "lora")
    if fine_tune_type != "full":
        linear_to_lora_layers(
            model,
            config.num_layers,
            config.lora_parameters,
            use_dora=(fine_tune_type == "dora"),
        )
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


def nparams(module):
    if hasattr(module, "bits"):
        n = 0 if not hasattr(module, "bias") else module.bias.size
        return n + module.weight.size * 32 // module.bits
    return sum(v.size for _, v in tree_flatten(module.parameters()))


def print_trainable_parameters(model):
    leaf_modules = tree_flatten(
        model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)
    )
    total_p = sum(nparams(m) for _, m in leaf_modules) / 10**6
    trainable_p = (
        sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    )
    print(
        f"Trainable parameters: {(trainable_p * 100 / total_p):.3f}% "
        f"({trainable_p:.3f}M/{total_p:.3f}M)"
    )
