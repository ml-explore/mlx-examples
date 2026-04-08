"""Convert a HuggingFace LLaDA model to MLX format.

Downloads (or reads from a local path) the LLaDA-8B-Base or LLaDA-8B-Instruct
model, remaps the weight names to match this MLX implementation, and saves the
result as a directory of safetensors files together with a config.json.

Usage:
    # Convert from HuggingFace Hub (requires huggingface_hub):
    python convert.py --hf-path GSAI-ML/LLaDA-8B-Instruct --mlx-path mlx_llada_instruct

    # Convert from a local HuggingFace checkpoint:
    python convert.py --hf-path /path/to/llada --mlx-path mlx_llada

    # Quantize weights to 4-bit while converting:
    python convert.py --hf-path GSAI-ML/LLaDA-8B-Instruct --mlx-path mlx_llada_4bit -q

Weight name mapping
-------------------
The LLaDA HuggingFace model uses the OLMo-derived naming convention under a
``transformer.*`` top-level namespace, with LLaMA-style block internals:

    transformer.wte.weight                          → embed_tokens.weight
    transformer.ln_f.weight                         → norm.weight
    transformer.ff_out.weight                       → lm_head.weight
    transformer.blocks.{i}.attn_norm.weight         → layers.{i}.input_layernorm.weight
    transformer.blocks.{i}.ff_norm.weight           → layers.{i}.post_attention_layernorm.weight
    transformer.blocks.{i}.q_proj.weight            → layers.{i}.self_attn.q_proj.weight
    transformer.blocks.{i}.k_proj.weight            → layers.{i}.self_attn.k_proj.weight
    transformer.blocks.{i}.v_proj.weight            → layers.{i}.self_attn.v_proj.weight
    transformer.blocks.{i}.attn_out.weight          → layers.{i}.self_attn.o_proj.weight
    transformer.blocks.{i}.ff_proj.weight           → layers.{i}.mlp.gate_proj.weight
    transformer.blocks.{i}.up_proj.weight           → layers.{i}.mlp.up_proj.weight
    transformer.blocks.{i}.ff_out.weight            → layers.{i}.mlp.down_proj.weight
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten


# ---------------------------------------------------------------------------
# Weight name mapping
# ---------------------------------------------------------------------------

# Simple prefix renames
_PREFIX_MAP = {
    "transformer.wte.weight": "embed_tokens.weight",
    "transformer.ln_f.weight": "norm.weight",
    "transformer.ff_out.weight": "lm_head.weight",
}

# Per-layer renames expressed as (source_suffix, dest_suffix)
_LAYER_MAP = [
    ("attn_norm.weight",                "input_layernorm.weight"),
    ("ff_norm.weight",                  "post_attention_layernorm.weight"),
    ("q_proj.weight",                   "self_attn.q_proj.weight"),
    ("k_proj.weight",                   "self_attn.k_proj.weight"),
    ("v_proj.weight",                   "self_attn.v_proj.weight"),
    ("attn_out.weight",                 "self_attn.o_proj.weight"),
    ("ff_proj.weight",                  "mlp.gate_proj.weight"),
    ("up_proj.weight",                  "mlp.up_proj.weight"),
    ("ff_out.weight",                   "mlp.down_proj.weight"),
]

_LAYER_RE = re.compile(r"^transformer\.blocks\.(\d+)\.(.+)$")


def _remap_key(key: str) -> str | None:
    """Return the MLX weight name for a HuggingFace weight name, or None to skip."""
    # Top-level renames
    if key in _PREFIX_MAP:
        return _PREFIX_MAP[key]

    # Per-layer renames
    m = _LAYER_RE.match(key)
    if m:
        layer_idx, suffix = m.group(1), m.group(2)
        for src_suffix, dst_suffix in _LAYER_MAP:
            if suffix == src_suffix:
                return f"layers.{layer_idx}.{dst_suffix}"

    # Drop everything else (rotary buffers, etc.)
    return None


# ---------------------------------------------------------------------------
# Config mapping
# ---------------------------------------------------------------------------


def _build_mlx_config(hf_config: dict) -> dict:
    """Build MLX ModelArgs-compatible config from a HuggingFace config dict."""
    return {
        "d_model": hf_config.get("d_model", 4096),
        "n_layers": hf_config.get("n_layers", 32),
        "n_heads": hf_config.get("n_heads", 32),
        "n_kv_heads": hf_config.get("n_kv_heads", hf_config.get("n_heads", 32)),
        "mlp_hidden_size": hf_config.get("mlp_hidden_size", 14336),
        "vocab_size": hf_config.get("vocab_size", hf_config.get("embedding_size", 126464)),
        "mask_token_id": hf_config.get("mask_token_id", 126336),
        "rms_norm_eps": hf_config.get("rms_norm_eps", 1e-5),
        "rope_theta": hf_config.get("rope_theta", 500000.0),
    }


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def convert(
    hf_path: str,
    mlx_path: str,
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
    dtype: str = "bfloat16",
    upload_repo: str | None = None,
):
    """Convert a HuggingFace LLaDA model to MLX format.

    Args:
        hf_path: HuggingFace Hub repo id (e.g. ``"GSAI-ML/LLaDA-8B-Instruct"``)
            or local directory containing the HuggingFace model files.
        mlx_path: Output directory for the converted model.
        quantize: If True, quantize weights with ``mlx.nn.quantize``.
        q_group_size: Quantization group size.
        q_bits: Quantization bit-width (4 or 8).
        dtype: Target float dtype (``"float16"`` or ``"bfloat16"``).
        upload_repo: If set, upload the converted model to this HuggingFace repo.
    """
    from transformers import AutoTokenizer

    hf_path = Path(hf_path)
    mlx_path = Path(mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    dtype_map = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
    target_dtype = dtype_map.get(dtype, mx.bfloat16)

    # ------------------------------------------------------------------ config
    config_path = hf_path / "config.json"
    if not config_path.exists():
        # Try downloading from Hub
        try:
            from huggingface_hub import snapshot_download
            hf_path = Path(snapshot_download(str(hf_path)))
            config_path = hf_path / "config.json"
        except ImportError:
            raise RuntimeError(
                "huggingface_hub not installed. "
                "Install it with: pip install huggingface_hub"
            )

    with open(config_path) as f:
        hf_config = json.load(f)

    mlx_config = _build_mlx_config(hf_config)
    with open(mlx_path / "config.json", "w") as f:
        json.dump(mlx_config, f, indent=2)
    print(f"Saved config → {mlx_path / 'config.json'}")

    # ----------------------------------------------------------------- weights
    import glob

    weight_files = sorted(glob.glob(str(hf_path / "*.safetensors")))
    if not weight_files:
        # Fall back to PyTorch bin files
        weight_files = sorted(glob.glob(str(hf_path / "*.bin")))

    if not weight_files:
        raise FileNotFoundError(f"No weight files found in {hf_path}")

    print(f"Loading weights from {len(weight_files)} file(s) …")
    raw_weights: dict[str, mx.array] = {}
    for wf in weight_files:
        if wf.endswith(".safetensors"):
            raw_weights.update(mx.load(wf).items())
        else:
            # PyTorch bin: load via numpy
            import torch
            state_dict = torch.load(wf, map_location="cpu")
            for k, v in state_dict.items():
                raw_weights[k] = mx.array(v.numpy())

    print(f"Loaded {len(raw_weights)} tensors. Remapping …")

    mlx_weights: dict[str, mx.array] = {}
    skipped: list[str] = []
    for src_key, tensor in raw_weights.items():
        dst_key = _remap_key(src_key)
        if dst_key is None:
            skipped.append(src_key)
            continue
        mlx_weights[dst_key] = tensor.astype(target_dtype)

    if skipped:
        print(f"Skipped {len(skipped)} tensors (rotary buffers, unused, …):")
        for k in skipped[:10]:
            print(f"  {k}")
        if len(skipped) > 10:
            print(f"  … and {len(skipped) - 10} more")

    print(f"Remapped {len(mlx_weights)} tensors.")

    # ---------------------------------------------------------------- quantize
    if quantize:
        from model import Model, ModelArgs

        model_args = ModelArgs(**mlx_config)
        model = Model(model_args)
        model.load_weights(list(mlx_weights.items()))
        mx.eval(model.parameters())

        nn.quantize(model, group_size=q_group_size, bits=q_bits)
        mx.eval(model.parameters())

        mlx_weights = dict(tree_flatten(model.parameters()))

        # Record quantization in config
        mlx_config["quantization"] = {"group_size": q_group_size, "bits": q_bits}
        with open(mlx_path / "config.json", "w") as f:
            json.dump(mlx_config, f, indent=2)
        print(f"Quantized to {q_bits}-bit (group_size={q_group_size}).")

    # ------------------------------------------------------------------- save
    # Split into ≤5 GB shards
    max_shard_bytes = 5 * 1024**3
    shard_weights: list[dict[str, mx.array]] = [{}]
    shard_bytes = 0

    for key, tensor in mlx_weights.items():
        nbytes = tensor.size * tensor.itemsize
        if shard_bytes + nbytes > max_shard_bytes and shard_weights[-1]:
            shard_weights.append({})
            shard_bytes = 0
        shard_weights[-1][key] = tensor
        shard_bytes += nbytes

    n_shards = len(shard_weights)
    for i, shard in enumerate(shard_weights):
        if n_shards == 1:
            out_file = mlx_path / "weights.safetensors"
        else:
            out_file = mlx_path / f"weights-{i + 1:05d}-of-{n_shards:05d}.safetensors"
        mx.save_safetensors(str(out_file), shard)
        print(f"Saved {out_file.name} ({len(shard)} tensors)")

    # ---------------------------------------------------------------- tokenizer
    print("Copying tokenizer files …")
    tok_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "added_tokens.json",
    ]
    for fname in tok_files:
        src = hf_path / fname
        if src.exists():
            shutil.copy2(src, mlx_path / fname)
            print(f"  {fname}")

    print(f"\nConversion complete → {mlx_path}")

    # ----------------------------------------------------------------- upload
    if upload_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(upload_repo, exist_ok=True)
        api.upload_folder(
            folder_path=str(mlx_path),
            repo_id=upload_repo,
            repo_type="model",
        )
        print(f"Uploaded to https://huggingface.co/{upload_repo}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace LLaDA model to MLX format"
    )
    parser.add_argument(
        "--hf-path", type=str, required=True,
        help="HuggingFace repo id or local path (e.g. GSAI-ML/LLaDA-8B-Instruct)",
    )
    parser.add_argument(
        "--mlx-path", type=str, required=True,
        help="Output directory for the converted MLX model",
    )
    parser.add_argument(
        "-q", "--quantize", action="store_true",
        help="Quantize model weights after conversion",
    )
    parser.add_argument(
        "--q-group-size", type=int, default=64,
        help="Quantization group size (default: 64)",
    )
    parser.add_argument(
        "--q-bits", type=int, default=4, choices=[4, 8],
        help="Quantization bits (default: 4)",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Weight dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--upload-repo", type=str, default=None,
        help="Upload the converted model to this HuggingFace repo id",
    )

    args = parser.parse_args()
    convert(
        hf_path=args.hf_path,
        mlx_path=args.mlx_path,
        quantize=args.quantize,
        q_group_size=args.q_group_size,
        q_bits=args.q_bits,
        dtype=args.dtype,
        upload_repo=args.upload_repo,
    )


if __name__ == "__main__":
    main()
