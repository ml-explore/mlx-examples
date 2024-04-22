# Copyright © 2023-2024 Apple Inc.

import argparse
import copy

import mlx.core as mx
import mlx.nn as nn
import models
import utils
from mlx.utils import tree_flatten


def quantize(weights, config, args):
    quantized_config = copy.deepcopy(config)

    # Load the model:
    model = models.Model(models.ModelArgs.from_dict(config))
    model.load_weights(list(weights.items()))

    # Quantize the model:
    nn.quantize(
        model,
        args.q_group_size,
        args.q_bits,
    )

    # Update the config:
    quantized_config["quantization"] = {
        "group_size": args.q_group_size,
        "bits": args.q_bits,
    }
    quantized_weights = dict(tree_flatten(model.parameters()))

    return quantized_weights, quantized_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face model to MLX format"
    )
    parser.add_argument(
        "--hf-path",
        type=str,
        help="Path to the Hugging Face model.",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx_model",
        help="Path to save the MLX model.",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        help="Generate a quantized model.",
        action="store_true",
    )
    parser.add_argument(
        "--q-group-size",
        help="Group size for quantization.",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--q-bits",
        help="Bits per weight for quantization.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--dtype",
        help="Type to save the parameters, ignored if -q is given.",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--upload-name",
        help="The name of model to upload to Hugging Face MLX Community",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    print("[INFO] Loading")
    weights, config, tokenizer = utils.fetch_from_hub(args.hf_path)

    dtype = mx.float16 if args.quantize else getattr(mx, args.dtype)
    weights = {k: v.astype(dtype) for k, v in weights.items()}
    if args.quantize:
        print("[INFO] Quantizing")
        weights, config = quantize(weights, config, args)

    utils.save_model(args.mlx_path, weights, tokenizer, config)
    if args.upload_name is not None:
        utils.upload_to_hub(args.mlx_path, args.upload_name, args.hf_path)
