from typing import List, Tuple
import os
import mlx.core as mx
from mlx.utils import tree_unflatten
from safetensors import safe_open
import torch
import numpy as np


def map_wan_weights(key: str, value: mx.array) -> List[Tuple[str, mx.array]]:
    # Remove .layers. from PyTorch Sequential to match MLX Python lists
    key = key.replace(".layers.", ".")
    
    # Handle conv transpose if needed
    if "patch_embedding.weight" in key:
        value = mx.transpose(value, (0, 2, 3, 4, 1))
    
    return [(key, value)]


def _flatten(params: List[List[Tuple[str, mx.array]]]) -> List[Tuple[str, mx.array]]:
    """Flatten nested list of parameter tuples"""
    return [(k, v) for p in params for (k, v) in p]


def load_wan_from_safetensors(
    safetensors_path: str, 
    model,
    float16: bool = False
):
    """
    Load WanModel weights from safetensors file(s) into MLX model.
    """
    import os
    import glob
    
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
        
        model.update(tree_unflatten(list(all_weights.items())))
    else:
        # Single file (1.3B model)
        print(f"Loading single file: {safetensors_path}")
        weights = mx.load(safetensors_path)
        model.update(tree_unflatten(list(weights.items())))
    
    print("WanModel weights loaded successfully!")
    return model


def convert_safetensors_to_mlx_weights(
    safetensors_path: str, 
    output_path: str,
    float16: bool = False
):
    """
    Convert safetensors file to MLX weights file.
    
    Args:
        safetensors_path: Input safetensors file
        output_path: Output MLX weights file (.safetensors)
        float16: Whether to use float16 precision
    """
    dtype = mx.float16 if float16 else mx.float32
    
    print(f"Converting safetensors to MLX format...")
    print(f"Input: {safetensors_path}")
    print(f"Output: {output_path}")
    print(f"Target dtype: {dtype}")
    
    # Load and convert weights
    weights = {}
    bfloat16_count = 0
    original_keys = []
    
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        original_keys = list(f.keys())  # Store keys before closing
        print(f"Processing {len(original_keys)} parameters...")
        
        for key in original_keys:
            tensor = f.get_tensor(key)
            
            # Handle BFloat16
            if tensor.dtype == torch.bfloat16:
                bfloat16_count += 1
                tensor = tensor.float()  # Convert to float32 first
            
            value = mx.array(tensor.numpy()).astype(dtype)
            
            # Apply mapping
            mapped = map_wan_weights(key, value)
            
            for new_key, new_value in mapped:
                weights[new_key] = new_value
    
    if bfloat16_count > 0:
        print(f"⚠️  Converted {bfloat16_count} BFloat16 tensors to {dtype}")
    
    # Print mapping summary
    skipped = len(original_keys) - len(weights)
    if skipped > 0:
        print(f"ℹ️  Skipped {skipped} activation layer parameters")
    
    # Save as MLX format
    print(f"Saving {len(weights)} parameters to: {output_path}")
    mx.save_safetensors(output_path, weights)
    
    # Print a few example keys for verification
    print("\nExample converted keys:")
    for i, key in enumerate(sorted(weights.keys())[:10]):
        print(f"  {key}: {weights[key].shape}")
    
    return weights

def convert_multiple_safetensors_to_mlx(
    checkpoint_dir: str,
    float16: bool = False
):
    """Convert multiple PyTorch safetensors files to MLX format."""
    import glob
    
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
        
        print(f"Converting {i}/{len(pytorch_files)}: {basename}")
        convert_safetensors_to_mlx_weights(pytorch_file, mlx_file, float16)
    
    print("All files converted successfully!")


def debug_weight_mapping(safetensors_path: str, float16: bool = False):
    """
    Debug function to see how weights are being mapped.
    """
    dtype = mx.float16 if float16 else mx.float32
    
    print("=== WAN Weight Mapping Debug ===")
    
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        # Check first 20 keys to see the mapping
        for i, key in enumerate(f.keys()):
            if i >= 20:
                break
                
            tensor = f.get_tensor(key)
            
            # Handle BFloat16
            original_dtype = tensor.dtype
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            
            value = mx.array(tensor.numpy()).astype(dtype)
            
            # Apply mapping
            mapped = map_wan_weights(key, value)
            
            if len(mapped) == 0:
                print(f"SKIPPED: {key} ({original_dtype})")
            elif len(mapped) == 1:
                new_key, new_value = mapped[0]
                if new_key == key:
                    print(f"DIRECT:  {key} ({original_dtype}) [{tensor.shape}]")
                else:
                    print(f"MAPPED:  {key} -> {new_key} [{tensor.shape}]")


def check_model_structure(model):
    """
    Print the structure of an MLX model to debug loading issues.
    """
    from mlx.utils import tree_flatten
    
    print("=== Model Structure ===")
    params = dict(tree_flatten(model))
    print(f"Model has {len(params)} parameters")
    
    print("\nFirst 20 parameter names:")
    for i, (key, value) in enumerate(params.items()):
        if i >= 20:
            break
        print(f"  {key}: {value.shape}")
    
    return params


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python wan_model_io.py <input.safetensors> <output.safetensors> [--fp16]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    use_fp16 = "--fp16" in sys.argv
    
    # Debug the mapping first (optional)
    debug_weight_mapping(input_file, use_fp16)
    
    # Convert weights
    convert_safetensors_to_mlx_weights(input_file, output_file, float16=use_fp16)
    
    print("Conversion complete!")