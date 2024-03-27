# Copyright © 2023 Apple Inc.

import argparse
from pathlib import Path

import mlx.core as mx
import utils
from mlx.utils import tree_flatten, tree_unflatten
from models.lora import LoRALinear

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--save-path",
        default="lora_fused_model",
        help="The path to save the fused model.",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        default="adapters.npz",
        help="Path to the trained adapter weights (npz or safetensors).",
    )

    args = parser.parse_args()

    model, tokenizer, config = utils.load(args.model)

    # Load adapters and get number of LoRA layers
    adapters = list(mx.load(args.adapter_file).items())
    lora_layers = len([m for m in adapters if "query.lora_a" in m[0]])

    # Freeze all layers other than LORA linears
    model.freeze()
    for block in model.encoder.blocks[len(model.encoder.blocks) - lora_layers :]:
        block.attn.query = LoRALinear.from_linear(block.attn.query)
        block.attn.value = LoRALinear.from_linear(block.attn.value)
    for block in model.decoder.blocks[len(model.decoder.blocks) - lora_layers :]:
        block.cross_attn.query = LoRALinear.from_linear(block.cross_attn.query)
        block.cross_attn.value = LoRALinear.from_linear(block.cross_attn.value)

    model.update(tree_unflatten(adapters))
    fused_linears = [
        (n, m.to_linear())
        for n, m in model.named_modules()
        if isinstance(m, LoRALinear)
    ]

    model.update_modules(tree_unflatten(fused_linears))
    weights = dict(tree_flatten(model.parameters()))
    utils.save_model(args.save_path, weights, tokenizer, config)
