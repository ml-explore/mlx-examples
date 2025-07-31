import json
from typing import Optional, List, Tuple
import mlx.core as mx
from mlx.utils import tree_unflatten
from safetensors import safe_open
import torch


def check_safetensors_dtypes(safetensors_path: str):
    """
    Check what dtypes are in the safetensors file.
    Useful for debugging dtype issues.
    """
    print(f"🔍 Checking dtypes in: {safetensors_path}")
    
    dtype_counts = {}
    
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            dtype_str = str(tensor.dtype)
            
            if dtype_str not in dtype_counts:
                dtype_counts[dtype_str] = []
            dtype_counts[dtype_str].append(key)
    
    print("📊 Dtype summary:")
    for dtype, keys in dtype_counts.items():
        print(f"  {dtype}: {len(keys)} parameters")
        if dtype == "torch.bfloat16":
            print(f"    ⚠️  BFloat16 detected - will convert to float32")
            print(f"    Examples: {keys[:3]}")
    
    return dtype_counts


def convert_tensor_dtype(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor to MLX-compatible dtype.
    """
    if tensor.dtype == torch.bfloat16:
        # Convert BFloat16 to float32
        return tensor.float()
    elif tensor.dtype == torch.float64:
        # Convert float64 to float32 for efficiency
        return tensor.float()
    else:
        # Keep other dtypes as-is
        return tensor


def map_t5_encoder_weights(key: str, value: mx.array) -> List[Tuple[str, mx.array]]:
    """
    Map T5 encoder weights from PyTorch format to MLX format.
    Following the pattern used in MLX Stable Diffusion.
    
    Args:
        key: Parameter name from PyTorch model
        value: Parameter tensor
        
    Returns:
        List of (key, value) tuples for MLX model
    """
    
    # Handle the main structural difference: FFN gate layer
    if ".ffn.gate.0.weight" in key:
        # PyTorch has Sequential(Linear, GELU) but MLX has separate gate_proj + gate_act
        key = key.replace(".ffn.gate.0.weight", ".ffn.gate_proj.weight")
        return [(key, value)]
    
    elif ".ffn.gate.0.bias" in key:
        # Handle bias if it exists
        key = key.replace(".ffn.gate.0.bias", ".ffn.gate_proj.bias")
        return [(key, value)]
    
    elif ".ffn.gate.1" in key:
        # Skip GELU activation parameters - MLX handles this separately
        print(f"Skipping GELU parameter: {key}")
        return []
    
    # Handle any other potential FFN mappings
    elif ".ffn.fc1.weight" in key:
        return [(key, value)]
    elif ".ffn.fc2.weight" in key:
        return [(key, value)]
    
    # Handle attention layers (should be direct mapping)
    elif ".attn.q.weight" in key:
        return [(key, value)]
    elif ".attn.k.weight" in key:
        return [(key, value)]
    elif ".attn.v.weight" in key:
        return [(key, value)]
    elif ".attn.o.weight" in key:
        return [(key, value)]
    
    # Handle embeddings and norms (direct mapping)
    elif "token_embedding.weight" in key:
        return [(key, value)]
    elif "pos_embedding.embedding.weight" in key:
        return [(key, value)]
    elif "norm1.weight" in key or "norm2.weight" in key or "norm.weight" in key:
        return [(key, value)]
    
    # Default: direct mapping for any other parameters
    else:
        return [(key, value)]


def _flatten(params: List[List[Tuple[str, mx.array]]]) -> List[Tuple[str, mx.array]]:
    """Flatten nested list of parameter tuples"""
    return [(k, v) for p in params for (k, v) in p]


def _load_safetensor_weights(
    mapper_func, 
    model, 
    weight_file: str, 
    float16: bool = False
):
    """
    Load safetensor weights using the mapping function.
    Based on MLX SD pattern.
    """
    dtype = mx.float16 if float16 else mx.float32
    
    # Load weights from safetensors file
    weights = {}
    with safe_open(weight_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            
            # Handle BFloat16 - convert to float32 first
            if tensor.dtype == torch.bfloat16:
                print(f"Converting BFloat16 to float32 for: {key}")
                tensor = tensor.float()  # Convert to float32
            
            weights[key] = mx.array(tensor.numpy()).astype(dtype)
    
    # Apply mapping function
    mapped_weights = _flatten([mapper_func(k, v) for k, v in weights.items()])
    
    # Update model with mapped weights
    model.update(tree_unflatten(mapped_weights))
    
    return model


def load_t5_encoder_from_safetensors(
    safetensors_path: str,
    model,  # Your MLX T5Encoder instance
    float16: bool = False
):
    """
    Load T5 encoder weights from safetensors file into MLX model.
    
    Args:
        safetensors_path: Path to the safetensors file
        model: Your MLX T5Encoder model instance
        float16: Whether to use float16 precision
        
    Returns:
        Model with loaded weights
    """
    print(f"Loading T5 encoder weights from: {safetensors_path}")
    
    # Load and map weights
    model = _load_safetensor_weights(
        map_t5_encoder_weights, 
        model, 
        safetensors_path, 
        float16
    )
    
    print("T5 encoder weights loaded successfully!")
    return model


def debug_weight_mapping(safetensors_path: str, float16: bool = False):
    """
    Debug function to see how weights are being mapped.
    Useful for troubleshooting conversion issues.
    """
    dtype = mx.float16 if float16 else mx.float32
    
    print("=== T5 Weight Mapping Debug ===")
    
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            
            # Handle BFloat16
            original_dtype = tensor.dtype
            if tensor.dtype == torch.bfloat16:
                print(f"Converting BFloat16 to float32 for: {key}")
                tensor = tensor.float()
            
            value = mx.array(tensor.numpy()).astype(dtype)
            
            # Apply mapping
            mapped = map_t5_encoder_weights(key, value)
            
            if len(mapped) == 0:
                print(f"SKIPPED: {key} ({original_dtype}) -> (no mapping)")
            elif len(mapped) == 1:
                new_key, new_value = mapped[0]
                if new_key == key:
                    print(f"DIRECT:  {key} ({original_dtype}) [{tensor.shape}]")
                else:
                    print(f"MAPPED:  {key} ({original_dtype}) -> {new_key} [{tensor.shape}]")
            else:
                print(f"SPLIT:   {key} ({original_dtype}) -> {len(mapped)} parameters")
                for new_key, new_value in mapped:
                    print(f"         -> {new_key} [{new_value.shape}]")


def convert_safetensors_to_mlx_weights(
    safetensors_path: str, 
    output_path: str,
    float16: bool = False
):
    """
    Convert safetensors file to MLX weights file (.npz format).
    
    Args:
        safetensors_path: Input safetensors file
        output_path: Output MLX weights file (.npz)
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
    
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            
            # Handle BFloat16
            # if tensor.dtype == torch.bfloat16:
                # bfloat16_count += 1
                # tensor = tensor.float()  # Convert to float32 first
            
            value = mx.array(tensor.numpy())#.astype(dtype)
            
            # Apply mapping
            mapped = map_t5_encoder_weights(key, value)
            
            for new_key, new_value in mapped:
                weights[new_key] = new_value
    
    if bfloat16_count > 0:
        print(f"⚠️  Converted {bfloat16_count} BFloat16 tensors to float32")
    
    # Save as MLX format
    print(f"Saving {len(weights)} parameters to: {output_path}")
    mx.save_safetensors(output_path, weights)
    
    return weights