import mlx.core as mx
from mlx.utils import tree_unflatten

from .linear import LoRALinear


def apply_lora_layers(model, adapter_file: str):
    adapters = list(mx.load(adapter_file).items())
    linear_replacements = {}
    lora_layers = set(
        [name.replace(".lora_a", "").replace(".lora_b", "") for name, _ in adapters]
    )

    for name, module in model.named_modules():
        if name in lora_layers:
            replacement_module = LoRALinear.from_linear(module)
            linear_replacements[name] = replacement_module

    model.update_modules(tree_unflatten(list(linear_replacements.items())))

    model.update(tree_unflatten(adapters))
    return model
