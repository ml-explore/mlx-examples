# Copyright © 2023-2024 Apple Inc.

import argparse
from enum import Enum

from .utils import convert, mixed_2_6, mixed_3_6


class MixedQuants(Enum):
    mixed_3_6 = "mixed_3_6"
    mixed_2_6 = "mixed_2_6"

    @classmethod
    def recipe_names(cls):
        return [member.name for member in cls]


def quant_args(arg):
    try:
        return MixedQuants[arg].value
    except KeyError:
        raise argparse.ArgumentTypeError(
            f"Invalid q-recipe {arg!r}. Choose from: {MixedQuants.recipe_names()}"
        )


def configure_parser() -> argparse.ArgumentParser:
    """
    Configures and returns the argument parser for the script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face model to MLX format"
    )

    parser.add_argument("--hf-path", type=str, help="Path to the Hugging Face model.")
    parser.add_argument(
        "--mlx-path", type=str, default="mlx_model", help="Path to save the MLX model."
    )
    parser.add_argument(
        "-q", "--quantize", help="Generate a quantized model.", action="store_true"
    )
    parser.add_argument(
        "--q-group-size", help="Group size for quantization.", type=int, default=64
    )
    parser.add_argument(
        "--q-bits", help="Bits per weight for quantization.", type=int, default=4
    )
    parser.add_argument(
        "--quant-predicate",
        help=f"Mixed-bit quantization recipe. Choices: {MixedQuants.recipe_names()}",
        type=quant_args,
        required=False,
    )
    parser.add_argument(
        "--dtype",
        help="Type to save the non-quantized parameters.",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-d",
        "--dequantize",
        help="Dequantize a quantized model.",
        action="store_true",
        default=False,
    )
    return parser


def main():
    parser = configure_parser()
    args = parser.parse_args()
    convert(**vars(args))


if __name__ == "__main__":
    main()
