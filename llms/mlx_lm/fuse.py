import argparse
import glob
import shutil
from pathlib import Path

from mlx.utils import tree_flatten, tree_unflatten

from .gguf import convert_to_gguf
from .tuner.dora import DoRALinear
from .tuner.lora import LoRALinear, LoRASwitchLinear
from .tuner.utils import apply_lora_layers, dequantize
from .utils import (
    fetch_from_hub,
    get_model_path,
    save_config,
    save_weights,
    upload_to_hub,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse fine-tuned adapters into the base model."
    )
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
        "--adapter-path",
        type=str,
        default="adapters",
        help="Path to the trained adapter weights and config.",
    )
    parser.add_argument(
        "--hf-path",
        type=str,
        default=None,
        help="Path to the original Hugging Face model. Required for upload if --model is a local directory.",
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--de-quantize",
        help="Generate a de-quantized model.",
        action="store_true",
    )
    parser.add_argument(
        "--export-gguf",
        help="Export model weights in GGUF format.",
        action="store_true",
    )
    parser.add_argument(
        "--gguf-path",
        help="Path to save the exported GGUF format model weights. Default is ggml-model-f16.gguf.",
        default="ggml-model-f16.gguf",
        type=str,
    )
    return parser.parse_args()


def main() -> None:
    print("Loading pretrained model")
    args = parse_arguments()

    model_path = get_model_path(args.model)
    model, config, tokenizer = fetch_from_hub(model_path)

    model.freeze()
    model = apply_lora_layers(model, args.adapter_path)

    fused_linears = [
        (n, m.to_linear())
        for n, m in model.named_modules()
        if isinstance(m, (LoRASwitchLinear, LoRALinear, DoRALinear))
    ]

    model.update_modules(tree_unflatten(fused_linears))

    if args.de_quantize:
        print("De-quantizing model")
        model = dequantize(model)

    weights = dict(tree_flatten(model.parameters()))

    save_path = Path(args.save_path)

    save_weights(save_path, weights)

    py_files = glob.glob(str(model_path / "*.py"))
    for file in py_files:
        shutil.copy(file, save_path)

    tokenizer.save_pretrained(save_path)

    if args.de_quantize:
        config.pop("quantization", None)

    save_config(config, config_path=save_path / "config.json")

    if args.export_gguf:
        model_type = config["model_type"]
        if model_type not in ["llama", "mixtral", "mistral"]:
            raise ValueError(
                f"Model type {model_type} not supported for GGUF conversion."
            )
        convert_to_gguf(model_path, weights, config, str(save_path / args.gguf_path))

    if args.upload_repo is not None:
        hf_path = args.hf_path or (
            args.model if not Path(args.model).exists() else None
        )
        if hf_path is None:
            raise ValueError(
                "Must provide original Hugging Face repo to upload local model."
            )
        upload_to_hub(args.save_path, args.upload_repo, hf_path)


if __name__ == "__main__":
    main()
