#!/usr/bin/env python3
"""
Convert DeciLM/Nemotron models to MLX format.
Handles NAS architecture with dummy layers and variable configurations.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model as load_hf_model
from mlx_lm.utils import save_model, get_model_path


def load_block_configs(config_path: Path) -> list:
    """Load block configurations from model config."""
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    block_configs = config.get("block_configs", [])
    if not block_configs:
        raise ValueError("No block_configs found in model config")
        
    return block_configs


def convert_attention_weights(hf_weights: Dict, layer_idx: int, block_config: dict) -> Dict:
    """Convert attention layer weights, handling dummy layers."""
    mlx_weights = {}
    attn_config = block_config["attention"]
    
    if attn_config.get("no_op", False):
        # Dummy attention - no weights
        return mlx_weights
        
    # Standard attention weight conversion
    prefix = f"model.layers.{layer_idx}.self_attn."
    mlx_prefix = f"model.layers.{layer_idx}.self_attn."
    
    # Convert projection weights
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        if f"{prefix}{proj}.weight" in hf_weights:
            mlx_weights[f"{mlx_prefix}{proj}.weight"] = hf_weights[f"{prefix}{proj}.weight"]
            
    return mlx_weights


def convert_ffn_weights(hf_weights: Dict, layer_idx: int, block_config: dict) -> Dict:
    """Convert FFN layer weights, handling dummy layers."""
    mlx_weights = {}
    ffn_config = block_config["ffn"]
    
    if ffn_config.get("no_op", False):
        # Dummy FFN - no weights
        return mlx_weights
        
    # Standard FFN weight conversion
    prefix = f"model.layers.{layer_idx}.mlp."
    mlx_prefix = f"model.layers.{layer_idx}.mlp."
    
    # Convert gate/up/down projections
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        if f"{prefix}{proj}.weight" in hf_weights:
            mlx_weights[f"{mlx_prefix}{proj}.weight"] = hf_weights[f"{prefix}{proj}.weight"]
            
    return mlx_weights


def convert_weights(hf_weights: Dict, block_configs: list) -> Dict:
    """Convert all model weights from HF to MLX format."""
    mlx_weights = {}
    
    # Convert embeddings
    if "model.embed_tokens.weight" in hf_weights:
        mlx_weights["model.embed_tokens.weight"] = hf_weights["model.embed_tokens.weight"]
        
    # Convert each layer based on its config
    for i, block_config in enumerate(block_configs):
        # Layer norms (always present)
        for norm in ["input_layernorm", "post_attention_layernorm"]:
            key = f"model.layers.{i}.{norm}.weight"
            if key in hf_weights:
                mlx_weights[key] = hf_weights[key]
                
        # Attention weights
        mlx_weights.update(convert_attention_weights(hf_weights, i, block_config))
        
        # FFN weights  
        mlx_weights.update(convert_ffn_weights(hf_weights, i, block_config))
        
    # Final norm and LM head
    if "model.norm.weight" in hf_weights:
        mlx_weights["model.norm.weight"] = hf_weights["model.norm.weight"]
    if "lm_head.weight" in hf_weights:
        mlx_weights["lm_head.weight"] = hf_weights["lm_head.weight"]
        
    return mlx_weights


def save_config(config_path: Path, hf_config: Dict, block_configs: list):
    """Save MLX model configuration."""
    mlx_config = {
        "model_type": "decilm",
        "hidden_size": hf_config["hidden_size"],
        "num_hidden_layers": hf_config["num_hidden_layers"],
        "intermediate_size": hf_config["intermediate_size"],
        "num_attention_heads": hf_config["num_attention_heads"],
        "num_key_value_heads": hf_config.get("num_key_value_heads", hf_config["num_attention_heads"]),
        "vocab_size": hf_config["vocab_size"],
        "rms_norm_eps": hf_config.get("rms_norm_eps", 1e-6),
        "rope_theta": hf_config.get("rope_theta", 10000),
        "rope_scaling": hf_config.get("rope_scaling"),
        "block_configs": block_configs,
    }
    
    with open(config_path, 'w') as f:
        json.dump(mlx_config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Convert DeciLM models to MLX")
    parser.add_argument(
        "--hf-path",
        type=str,
        required=True,
        help="Path to HuggingFace model or repo ID",
    )
    parser.add_argument(
        "--mlx-path", 
        type=str,
        required=True,
        help="Output path for MLX model",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize the model",
    )
    parser.add_argument(
        "--q-bits",
        type=int,
        default=4,
        help="Number of bits for quantization",
    )
    parser.add_argument(
        "--q-group-size",
        type=int,
        default=64,
        help="Group size for quantization",
    )
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.hf_path}")
    model_path = get_model_path(args.hf_path)
    
    # Load configurations
    hf_config_path = model_path / "config.json"
    with open(hf_config_path, 'r') as f:
        hf_config = json.load(f)
        
    block_configs = hf_config.get("block_configs", [])
    if not block_configs:
        raise ValueError("This doesn't appear to be a DeciLM model (no block_configs)")
        
    print(f"Found {len(block_configs)} blocks with NAS configuration")
    
    # Count dummy layers
    dummy_attn = sum(1 for bc in block_configs if bc["attention"].get("no_op", False))
    dummy_ffn = sum(1 for bc in block_configs if bc["ffn"].get("no_op", False))
    print(f"Dummy layers: {dummy_attn} attention, {dummy_ffn} FFN")
    
    # Load HF weights
    print("Loading weights...")
    model, _ = load_hf_model(args.hf_path)
    hf_weights = dict(model.state_dict())
    
    # Convert weights
    print("Converting weights to MLX format...")
    mlx_weights = convert_weights(hf_weights, block_configs)
    
    # Quantize if requested
    if args.quantize:
        print(f"Quantizing to {args.q_bits} bits...")
        mlx_weights = mx.quantize(
            mlx_weights, 
            bits=args.q_bits,
            group_size=args.q_group_size
        )
        
    # Save model
    output_path = Path(args.mlx_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to {output_path}")
    
    # Save weights
    mx.save_safetensors(str(output_path / "model.safetensors"), mlx_weights)
    
    # Save config
    save_config(output_path / "config.json", hf_config, block_configs)
    
    # Copy tokenizer files
    for file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = model_path / file
        if src.exists():
            shutil.copy(src, output_path / file)
            
    print("Conversion complete!")
    

if __name__ == "__main__":
    main()