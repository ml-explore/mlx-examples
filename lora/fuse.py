# Copyright © 2023 Apple Inc.

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
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
    parser.add_argument(
        "--hf-path",
        help=(
            "Path to the original Hugging Face model. This is "
            "required for upload if --model is a local directory."
        ),
        type=str,
        default=None,
    )
    parser.add_argument(
        "--upload-name",
        help="The name of model to upload to Hugging Face MLX Community",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-d",
        "--de-quantize",
        help="Generate a de-quantized model.",
        action="store_true",
    )

    print("Loading pretrained model")
    args = parser.parse_args()

    model, tokenizer, config = utils.load(args.model)

    # Load adapters and get number of LoRA layers
    adapters = list(mx.load(args.adapter_file).items())
    lora_layers = len([m for m in adapters if "q_proj.lora_a" in m[0]])

    # Freeze all layers other than LORA linears
    model.freeze()
    for l in model.model.layers[len(model.model.layers) - lora_layers :]:
        l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj)
        l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj)
        if hasattr(l, "block_sparse_moe"):
            l.block_sparse_moe.gate = LoRALinear.from_linear(l.block_sparse_moe.gate)

    model.update(tree_unflatten(adapters))
    fused_linears = [
        (n, m.to_linear())
        for n, m in model.named_modules()
        if isinstance(m, LoRALinear)
    ]

    model.update_modules(tree_unflatten(fused_linears))

    if args.de_quantize:
        de_quantize_layers = []
        for n, m in model.named_modules():
            if isinstance(m, nn.QuantizedLinear):
                bias = "bias" in m
                weight = m.weight
                weight = mx.dequantize(
                    weight,
                    m.scales,
                    m.biases,
                    m.group_size,
                    m.bits,
                ).astype(mx.float16)
                output_dims, input_dims = weight.shape
                linear = nn.Linear(input_dims, output_dims, bias=bias)
                linear.weight = weight
                if bias:
                    linear.bias = m.bias
                de_quantize_layers.append((n, linear))

        model.update_modules(tree_unflatten(de_quantize_layers))

    weights = dict(tree_flatten(model.parameters()))
    if args.de_quantize:
        config.pop("quantization", None)
    utils.save_model(args.save_path, weights, tokenizer, config)

    if args.upload_name is not None:
        hf_path = args.hf_path
        if not Path(args.model).exists():
            # If the model path doesn't exist, assume it's an HF repo
            hf_path = args.model
        elif hf_path is None:
            raise ValueError(
                "Must provide original Hugging Face repo to upload local model."
            )
        utils.upload_to_hub(args.save_path, args.upload_name, hf_path)
