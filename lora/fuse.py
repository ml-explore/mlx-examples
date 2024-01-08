# Copyright © 2023 Apple Inc.

import argparse

import mlx.core as mx
import models
import utils
from mlx.utils import tree_flatten, tree_unflatten

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory.",
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
        help="Save/load path for the trained adapter weights.",
    )
    parser.add_argument(
        "--hf-path",
        help="Path to the original Hugging Face model (required for upload).",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--upload-name",
        help="The name of model to upload to Hugging Face MLX Community",
        type=str,
        default=None,
    )

    print("Loading pretrained model")
    args = parser.parse_args()

    model, tokenizer, config = models.load(args.model)

    # Load adapters and get number of LoRA layers
    adapters = list(mx.load(args.adapter_file).items())
    lora_layers = len([m for m in adapters if "q_proj.lora_a" in m[0]])

    # Freeze all layers other than LORA linears
    model.freeze()
    for l in model.model.layers[-lora_layers:]:
        l.self_attn.q_proj = models.LoRALinear.from_linear(l.self_attn.q_proj)
        l.self_attn.v_proj = models.LoRALinear.from_linear(l.self_attn.v_proj)

    model.update(tree_unflatten(adapters))
    fused_linears = [
        (n, m.to_linear())
        for n, m in model.named_modules()
        if isinstance(m, models.LoRALinear)
    ]

    model.update_modules(tree_unflatten(fused_linears))
    weights = dict(tree_flatten(model.parameters()))
    utils.save_model(args.save_path, weights, tokenizer._tokenizer, config)
    if args.upload_name is not None:
        if args.hf_path is None:
            raise ValueError("Must provide original Hugging Face repo to upload model")
        utils.upload_model(args.save_path, args.upload_name, args.hf_path)
