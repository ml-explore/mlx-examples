import os

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from .lora import LoRALinear


def linear_to_lora_layers(model: nn.Module, num_lora_layers: int):
    """
    Convert some of the models linear layers to lora layers.

    Args:
        model (nn.Module): The neural network model.
        num_lora_layers (int): The number of blocks to convert to lora layers
        starting from the last layer.
    """

    def check_lora_layers(num_model):
        if num_lora_layers > num_model:
            raise ValueError(
                f"Requested {num_lora_layers} LoRA layers "
                f"but the model only has {num_model} layers."
            )

    if model.model_type in [
        "mistral",
        "llama",
        "phi",
        "mixtral",
        "stablelm_epoch",
        "qwen2",
        "gemma",
    ]:
        check_lora_layers(len(model.model.layers))

        for l in model.model.layers[len(model.model.layers) - num_lora_layers :]:
            l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj)
            l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj)
            if hasattr(l, "block_sparse_moe"):
                l.block_sparse_moe.gate = LoRALinear.from_linear(
                    l.block_sparse_moe.gate
                )
    elif model.model_type == "olmo":
        check_lora_layers(len(model.model.transformer.blocks))

        for l in model.model.transformer.blocks[
            len(model.model.transformer.blocks) - num_lora_layers :
        ]:
            l.att_proj = LoRALinear.from_linear(l.att_proj)
    elif model.model_type == "phi-msft":
        check_lora_layers(len(model.transformer.h))

        for l in model.transformer.h[len(model.transformer.h) - num_lora_layers :]:
            l.mixer.Wqkv = LoRALinear.from_linear(l.mixer.Wqkv)
            l.moe.gate = LoRALinear.from_linear(l.moe.gate)

    else:
        raise ValueError(f"Lora does not support {model.model_type}")


def apply_lora_layers(model: nn.Module, adapter_file: str) -> nn.Module:
    """
    Apply LoRA layers to the model.

    Args:
        model (nn.Module): The neural network model.
        adapter_file (str): Path to the adapter configuration file.

    Returns:
        nn.Module: The updated model with LoRA layers applied.
    """
    if not os.path.exists(adapter_file):
        raise FileNotFoundError(f"The adapter file does not exist: {adapter_file}")

    adapters = list(mx.load(adapter_file).items())

    linear_replacements = []
    lora_layers = set(
        [name.replace(".lora_a", "").replace(".lora_b", "") for name, _ in adapters]
    )
    for name, module in model.named_modules():
        if name in lora_layers:
            replacement_module = LoRALinear.from_linear(module)
            linear_replacements.append((name, replacement_module))

    model.update_modules(tree_unflatten(linear_replacements))

    model.update(tree_unflatten(adapters))

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
