import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

import mlx.core as mx
import torch
from huggingface_hub import snapshot_download


def download(hf_repo: str) -> Path:
    """Download model from Hugging Face."""
    return Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=["*.safetensors", "*.json", "*.bin", "*.txt"],
        )
    )


def remap_key(key: str) -> str:
    """Remap HuggingFace ESM key names to MLX format."""

    # Skip position embeddings and position_ids
    if "position_embeddings" in key or "position_ids" in key:
        return None

    # Map lm_head components properly
    if key == "lm_head.decoder.weight":
        return "lm_head.weight"
    if key == "lm_head.decoder.bias":
        return "lm_head.bias"
    if key == "lm_head.dense.weight":
        return "lm_head.dense.weight"
    if key == "lm_head.dense.bias":
        return "lm_head.dense.bias"
    if key == "lm_head.layer_norm.weight":
        return "lm_head.layer_norm.weight"
    if key == "lm_head.layer_norm.bias":
        return "lm_head.layer_norm.bias"

    # Core remapping patterns
    key = key.replace("esm.embeddings.word_embeddings", "embed_tokens")
    key = key.replace("esm.encoder.emb_layer_norm_after", "emb_layer_norm_after")
    key = key.replace("esm.encoder.layer.", "layer_")
    key = key.replace("esm.contact_head", "contact_head")
    key = key.replace("lm_head", "lm_head")

    # Attention patterns
    key = key.replace(".attention.self.", ".self_attn.")
    key = key.replace(".attention.output.dense", ".self_attn.out_proj")
    key = key.replace(".attention.LayerNorm", ".self_attn_layer_norm")
    key = key.replace(".query", ".q_proj")
    key = key.replace(".key", ".k_proj")
    key = key.replace(".value", ".v_proj")
    key = key.replace(".rotary_embeddings", ".rot_emb")

    # FFN patterns
    key = key.replace(".intermediate.dense", ".fc1")
    key = key.replace(".output.dense", ".fc2")
    key = key.replace(".LayerNorm", ".final_layer_norm")

    return key


def load_weights(model_path: Path) -> Dict:
    """Load weights from safetensors or PyTorch bin files."""

    # Check for safetensors file
    safetensors_path = model_path / "model.safetensors"
    if safetensors_path.exists():
        print("Loading from safetensors...")
        return mx.load(str(safetensors_path))

    # Check for single bin file
    single_bin_path = model_path / "pytorch_model.bin"
    if single_bin_path.exists():
        print("Loading from pytorch_model.bin...")
        state_dict = torch.load(str(single_bin_path), map_location="cpu")
        return {k: v.numpy() for k, v in state_dict.items()}

    # Check for sharded bin files
    index_file = model_path / "pytorch_model.bin.index.json"
    if index_file.exists():
        print("Loading from sharded bin files...")
        with open(index_file) as f:
            index = json.load(f)

        # Get unique shard files
        shard_files = set(index["weight_map"].values())

        # Load all shards
        state_dict = {}
        for shard_file in sorted(shard_files):
            print(f"  Loading shard: {shard_file}")
            shard_path = model_path / shard_file
            shard_dict = torch.load(str(shard_path), map_location="cpu")
            state_dict.update(shard_dict)

        return {k: v.numpy() for k, v in state_dict.items()}

    raise ValueError(f"No model weights found in {model_path}")


def convert(model_path: Path) -> Dict[str, mx.array]:
    """Convert ESM weights to MLX format."""

    # Load weights from any format
    weights = load_weights(model_path)

    # Convert keys and create MLX arrays
    mlx_weights = {}
    for key, value in weights.items():
        mlx_key = remap_key(key)
        if mlx_key is not None:
            mlx_weights[mlx_key] = (
                mx.array(value) if not isinstance(value, mx.array) else value
            )

    # If lm_head.weight is missing but embed_tokens.weight exists, set up weight sharing
    # (This is for smaller models that don't have a separate lm_head.decoder.weight)
    if "lm_head.weight" not in mlx_weights and "embed_tokens.weight" in mlx_weights:
        mlx_weights["lm_head.weight"] = mlx_weights["embed_tokens.weight"]

    return mlx_weights


def main():
    parser = argparse.ArgumentParser(description="Convert ESM weights to MLX format")
    parser.add_argument(
        "--hf-path", default="facebook/esm2_t6_8M_UR50D", help="Hugging Face model path"
    )
    parser.add_argument("--mlx-path", default=None, help="Output path for MLX model")
    parser.add_argument(
        "--checkpoints-dir",
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)",
    )

    args = parser.parse_args()

    # Download model
    print(f"Downloading {args.hf_path}...")
    model_path = download(args.hf_path)

    # Set output path
    if args.mlx_path is None:
        model_name = args.hf_path.split("/")[-1]
        checkpoints_dir = Path(args.checkpoints_dir)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        args.mlx_path = checkpoints_dir / f"mlx-{model_name}"
    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    # Convert weights
    print("Converting weights...")
    mlx_weights = convert(model_path)

    # Save weights
    print(f"Saving MLX weights to {mlx_path}...")
    mx.save_safetensors(str(mlx_path / "model.safetensors"), mlx_weights)

    # Copy config and other files
    print("Copying config...")
    shutil.copy(model_path / "config.json", mlx_path / "config.json")

    for file_name in ["special_tokens_map.json", "tokenizer.json", "vocab.txt"]:
        src_file = model_path / file_name
        if src_file.exists():
            shutil.copy(src_file, mlx_path / file_name)

    print(f"Conversion complete! MLX model saved to {mlx_path}")


if __name__ == "__main__":
    main()
