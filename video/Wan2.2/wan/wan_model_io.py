# wan_model_io.py

from typing import List, Tuple, Set, Dict
import os
import mlx.core as mx
from mlx.utils import tree_unflatten, tree_flatten
from safetensors import safe_open
import torch
import numpy as np
import glob


def map_wan_2_2_weights(key: str, value: mx.array) -> List[Tuple[str, mx.array]]:
    """Map PyTorch WAN 2.2 weights to MLX format."""
    
    # Only add .layers to Sequential WITHIN components, not to blocks themselves
    # blocks.N stays as blocks.N (not blocks.layers.N)
    
    # Handle Sequential layers - PyTorch uses .0, .1, .2, MLX uses .layers.0, .layers.1, .layers.2
    # Only for components INSIDE blocks and top-level modules
    if ".ffn." in key and not ".layers." in key:
        # Replace .ffn.0 with .ffn.layers.0, etc.
        key = key.replace(".ffn.0.", ".ffn.layers.0.")
        key = key.replace(".ffn.1.", ".ffn.layers.1.")
        key = key.replace(".ffn.2.", ".ffn.layers.2.")
    
    if "text_embedding." in key and not ".layers." in key:
        for i in range(10):
            key = key.replace(f"text_embedding.{i}.", f"text_embedding.layers.{i}.")
    
    if "time_embedding." in key and not ".layers." in key:
        for i in range(10):
            key = key.replace(f"time_embedding.{i}.", f"time_embedding.layers.{i}.")
    
    if "time_projection." in key and not ".layers." in key:
        for i in range(10):
            key = key.replace(f"time_projection.{i}.", f"time_projection.layers.{i}.")
    
    # Handle conv transpose for patch_embedding
    if "patch_embedding.weight" in key:
        # PyTorch Conv3d: (out_channels, in_channels, D, H, W)
        # MLX Conv3d: (out_channels, D, H, W, in_channels)
        value = mx.transpose(value, (0, 2, 3, 4, 1))
    
    return [(key, value)]


def check_parameter_mismatch(model, weights: Dict[str, mx.array]) -> Tuple[Set[str], Set[str]]:
    """
    Check for parameter mismatches between model and weights.
    
    Returns:
        (model_only, weights_only): Sets of parameter names that exist only in model or weights
    """
    # Get all parameter names from model
    model_params = dict(tree_flatten(model.parameters()))
    model_keys = set(model_params.keys())
    
    # Remove computed buffers that aren't loaded from weights
    computed_buffers = {'freqs'}  # Add any other computed buffers here
    model_keys = model_keys - computed_buffers
    
    # Get all parameter names from weights
    weight_keys = set(weights.keys())
    
    # Find differences
    model_only = model_keys - weight_keys
    weights_only = weight_keys - model_keys
    
    return model_only, weights_only


def load_wan_2_2_from_safetensors(
    safetensors_path: str, 
    model,
    float16: bool = False,
    check_mismatch: bool = True
):
    """
    Load WAN 2.2 Model weights from safetensors file(s) into MLX model.
    
    Args:
        safetensors_path: Path to safetensors file or directory
        model: MLX model instance
        float16: Whether to use float16 precision
        check_mismatch: Whether to check for parameter mismatches
    """
    if os.path.isdir(safetensors_path):
        # Multiple files (14B model) - only diffusion_mlx_model files
        pattern = os.path.join(safetensors_path, "diffusion_mlx_model*.safetensors")
        safetensor_files = sorted(glob.glob(pattern))
        print(f"Found {len(safetensor_files)} diffusion_mlx_model safetensors files")
        
        # Load all files and merge weights
        all_weights = {}
        for file_path in safetensor_files:
            print(f"Loading: {file_path}")
            weights = mx.load(file_path)
            all_weights.update(weights)
        
        if check_mismatch:
            model_only, weights_only = check_parameter_mismatch(model, all_weights)
            
            if model_only:
                print(f"\n⚠️  WARNING: {len(model_only)} parameters in model but NOT in weights:")
                for param in sorted(model_only)[:10]:  # Show first 10
                    print(f"  - {param}")
                if len(model_only) > 10:
                    print(f"  ... and {len(model_only) - 10} more")
            
            if weights_only:
                print(f"\n⚠️  WARNING: {len(weights_only)} parameters in weights but NOT in model:")
                for param in sorted(weights_only)[:10]:  # Show first 10
                    print(f"  - {param}")
                if len(weights_only) > 10:
                    print(f"  ... and {len(weights_only) - 10} more")
            
            if not model_only and not weights_only:
                print("\n✅ Perfect match: All parameters align between model and weights!")
        
        model.update(tree_unflatten(list(all_weights.items())))
    else:
        # Single file
        print(f"Loading single file: {safetensors_path}")
        weights = mx.load(safetensors_path)
        
        if check_mismatch:
            model_only, weights_only = check_parameter_mismatch(model, weights)
            
            if model_only:
                print(f"\n⚠️  WARNING: {len(model_only)} parameters in model but NOT in weights:")
                for param in sorted(model_only)[:10]:  # Show first 10
                    print(f"  - {param}")
                if len(model_only) > 10:
                    print(f"  ... and {len(model_only) - 10} more")
            
            if weights_only:
                print(f"\n⚠️  WARNING: {len(weights_only)} parameters in weights but NOT in model:")
                for param in sorted(weights_only)[:10]:  # Show first 10
                    print(f"  - {param}")
                if len(weights_only) > 10:
                    print(f"  ... and {len(weights_only) - 10} more")
            
            if not model_only and not weights_only:
                print("\n✅ Perfect match: All parameters align between model and weights!")
        
        model.update(tree_unflatten(list(weights.items())))
    
    print("\nWAN 2.2 Model weights loaded successfully!")
    return model


def convert_wan_2_2_safetensors_to_mlx(
    safetensors_path: str, 
    output_path: str,
    float16: bool = False,
    model=None  # Optional: provide model instance to check parameter alignment
):
    """
    Convert WAN 2.2 PyTorch safetensors file to MLX weights file.
    
    Args:
        safetensors_path: Input safetensors file
        output_path: Output MLX weights file (.safetensors)
        float16: Whether to use float16 precision
        model: Optional MLX model instance to check parameter alignment
    """
    dtype = mx.float16 if float16 else mx.float32
    
    print(f"Converting WAN 2.2 safetensors to MLX format...")
    print(f"Input: {safetensors_path}")
    print(f"Output: {output_path}")
    print(f"Target dtype: {dtype}")
    
    # Load and convert weights
    weights = {}
    bfloat16_count = 0
    
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"Processing {len(keys)} parameters...")
        
        for key in keys:
            tensor = f.get_tensor(key)
            
            # Handle BFloat16
            if tensor.dtype == torch.bfloat16:
                bfloat16_count += 1
                tensor = tensor.float()  # Convert to float32 first
            
            value = mx.array(tensor.numpy()).astype(dtype)
            
            # Apply mapping
            mapped = map_wan_2_2_weights(key, value)
            
            for new_key, new_value in mapped:
                weights[new_key] = new_value
    
    if bfloat16_count > 0:
        print(f"⚠️  Converted {bfloat16_count} BFloat16 tensors to {dtype}")
    
    # Check parameter alignment if model provided
    if model is not None:
        print("\nChecking parameter alignment with model...")
        model_only, weights_only = check_parameter_mismatch(model, weights)
        
        if model_only:
            print(f"\n⚠️  WARNING: {len(model_only)} parameters in model but NOT in converted weights:")
            for param in sorted(model_only)[:10]:
                print(f"  - {param}")
            if len(model_only) > 10:
                print(f"  ... and {len(model_only) - 10} more")
        
        if weights_only:
            print(f"\n⚠️  WARNING: {len(weights_only)} parameters in converted weights but NOT in model:")
            for param in sorted(weights_only)[:10]:
                print(f"  - {param}")
            if len(weights_only) > 10:
                print(f"  ... and {len(weights_only) - 10} more")
        
        if not model_only and not weights_only:
            print("\n✅ Perfect match: All parameters align between model and converted weights!")
    
    # Save as MLX format
    print(f"\nSaving {len(weights)} parameters to: {output_path}")
    mx.save_safetensors(output_path, weights)
    
    # Print a few example keys for verification
    print("\nExample converted keys:")
    for i, key in enumerate(sorted(weights.keys())[:10]):
        print(f"  {key}: {weights[key].shape}")
    
    return weights


def convert_multiple_wan_2_2_safetensors_to_mlx(
    checkpoint_dir: str,
    float16: bool = False
):
    """Convert multiple WAN 2.2 PyTorch safetensors files to MLX format."""
    # Find all PyTorch model files
    pytorch_pattern = os.path.join(checkpoint_dir, "diffusion_pytorch_model-*.safetensors")
    pytorch_files = sorted(glob.glob(pytorch_pattern))
    
    if not pytorch_files:
        raise FileNotFoundError(f"No PyTorch model files found matching: {pytorch_pattern}")
    
    print(f"Converting {len(pytorch_files)} PyTorch files to MLX format...")
    
    for i, pytorch_file in enumerate(pytorch_files, 1):
        # Extract the suffix (e.g., "00001-of-00006")
        basename = os.path.basename(pytorch_file)
        suffix = basename.replace("diffusion_pytorch_model-", "").replace(".safetensors", "")
        
        # Create MLX filename
        mlx_file = os.path.join(checkpoint_dir, f"diffusion_mlx_model-{suffix}.safetensors")
        
        print(f"\nConverting {i}/{len(pytorch_files)}: {basename}")
        convert_wan_2_2_safetensors_to_mlx(pytorch_file, mlx_file, float16)
    
    print("\nAll files converted successfully!")


def debug_wan_2_2_weight_mapping(safetensors_path: str, float16: bool = False):
    """
    Debug function to see how WAN 2.2 weights are being mapped.
    """
    dtype = mx.float16 if float16 else mx.float32
    
    print("=== WAN 2.2 Weight Mapping Debug ===")
    
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        # Check first 30 keys to see the mapping
        for i, key in enumerate(f.keys()):
            if i >= 30:
                break
                
            tensor = f.get_tensor(key)
            
            # Handle BFloat16
            original_dtype = tensor.dtype
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            
            value = mx.array(tensor.numpy()).astype(dtype)
            
            # Apply mapping
            mapped = map_wan_2_2_weights(key, value)
            
            new_key, new_value = mapped[0]
            if new_key == key:
                print(f"UNCHANGED: {key} [{tensor.shape}]")
            else:
                print(f"MAPPED:    {key} -> {new_key} [{tensor.shape}]")