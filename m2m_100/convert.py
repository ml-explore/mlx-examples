import argparse

import numpy
from transformers import M2M100ForConditionalGeneration


def convert(model_name: str, mlx_model: str) -> None:
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    # save the tensors
    tensors = {
        key: tensor.numpy() for key, tensor in model.state_dict().items()
    }
    numpy.savez(mlx_model, **tensors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert M2M style model weights to MLX.")
    parser.add_argument(
        "--nllb-model",
        type=str,
        default="facebook/nllb-200-1.3B",
        help="The huggingface name of the NLLB model to save",
    )
    parser.add_argument(
        "--mlx-model",
        type=str,
        default="weights/nllb-200-1.3B.npz",
        help="The output path for the MLX weights.",
    )
    args = parser.parse_args()

    convert(args.nllb_model, args.mlx_model)
