import torch
import mlx.core as mx
import numpy as np
from typing import Dict, Tuple
from safetensors import safe_open

def convert_pytorch_to_mlx(pytorch_path: str, output_path: str, float16: bool = False):
    """
    Convert PyTorch VAE weights to MLX format with correct mapping.
    """
    print(f"Converting {pytorch_path} -> {output_path}")
    dtype = mx.float16 if float16 else mx.float32
    
    # Load PyTorch weights
    if pytorch_path.endswith('.safetensors'):
        weights = {}
        with safe_open(pytorch_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.float()
                weights[key] = tensor.numpy()
    else:
        checkpoint = torch.load(pytorch_path, map_location='cpu')
        weights = {}
        state_dict = checkpoint if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint else checkpoint.get('state_dict', checkpoint)
        for key, tensor in state_dict.items():
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            weights[key] = tensor.numpy()
    
    # Convert weights
    mlx_weights = {}
    
    for key, value in weights.items():
        # Skip these
        if any(skip in key for skip in ["num_batches_tracked", "running_mean", "running_var"]):
            continue
            
        # Convert weight formats
        if value.ndim == 5 and "weight" in key:  # Conv3d weights
            # PyTorch: (out_channels, in_channels, D, H, W)
            # MLX: (out_channels, D, H, W, in_channels)
            value = np.transpose(value, (0, 2, 3, 4, 1))
        elif value.ndim == 4 and "weight" in key:  # Conv2d weights
            # PyTorch: (out_channels, in_channels, H, W)
            # MLX Conv2d expects: (out_channels, H, W, in_channels)
            value = np.transpose(value, (0, 2, 3, 1))
        elif value.ndim == 1 and "bias" in key:  # Conv biases
            # Keep as is - MLX uses same format
            pass
        
        # Map the key
        new_key = key
        
        # Map residual block internals within Sequential
        # PyTorch: encoder.downsamples.0.residual.0.gamma
        # MLX: encoder.downsamples.layers.0.residual.layers.0.gamma
        import re
        
        # Add .layers to Sequential modules
        new_key = re.sub(r'\.downsamples\.(\d+)', r'.downsamples.layers.\1', new_key)
        new_key = re.sub(r'\.upsamples\.(\d+)', r'.upsamples.layers.\1', new_key)
        new_key = re.sub(r'\.middle\.(\d+)', r'.middle.layers.\1', new_key)
        new_key = re.sub(r'\.head\.(\d+)', r'.head.layers.\1', new_key)
        
        # Map residual Sequential internals
        if ".residual." in new_key:
            match = re.search(r'\.residual\.(\d+)\.', new_key)
            if match:
                idx = int(match.group(1))
                if idx == 0:  # First RMS_norm
                    new_key = re.sub(r'\.residual\.0\.', '.residual.layers.0.', new_key)
                elif idx == 1:  # SiLU - skip
                    continue
                elif idx == 2:  # First Conv3d
                    new_key = re.sub(r'\.residual\.2\.', '.residual.layers.2.', new_key)
                elif idx == 3:  # Second RMS_norm
                    new_key = re.sub(r'\.residual\.3\.', '.residual.layers.3.', new_key)
                elif idx == 4:  # Second SiLU - skip
                    continue
                elif idx == 5:  # Dropout - could be Identity in MLX
                    if "Dropout" in key:
                        continue
                    new_key = re.sub(r'\.residual\.5\.', '.residual.layers.5.', new_key)
                elif idx == 6:  # Second Conv3d
                    new_key = re.sub(r'\.residual\.6\.', '.residual.layers.6.', new_key)
        
        # Map resample internals
        if ".resample." in new_key:
            # In both Encoder and Decoder Resample blocks, the Conv2d is at index 1
            # in the nn.Sequential block, following either a padding or upsample layer.
            # We just need to map PyTorch's .1 to MLX's .layers.1
            if ".resample.1." in new_key:
                new_key = new_key.replace(".resample.1.", ".resample.layers.1.")
            
            # The layers at index 0 (ZeroPad2d, Upsample) have no weights, so we can
            # safely skip any keys associated with them.
            if ".resample.0." in key:
                continue
        
        # Map head internals (already using Sequential in MLX)
        # Just need to handle the layers index
        
        # Handle shortcut layers
        if ".shortcut." in new_key and "Identity" not in key:
            # Shortcut Conv3d layers - keep as is
            pass
        elif "Identity" in key:
            # Skip Identity modules
            continue
        
        # Handle time_conv in Resample
        if "time_conv" in new_key:
            # Keep as is - already correctly named
            pass
        
        # Handle attention layers
        if "to_qkv" in new_key or "proj" in new_key:
            # Keep as is - already correctly named
            pass
        
       # In the conversion script
        if "gamma" in new_key:
            # Squeeze gamma from (C, 1, 1) or (C, 1, 1, 1) to just (C,)
            value = np.squeeze(value)  # This removes all dimensions of size 1
            # Result will always be 1D array of shape (C,)
        
        # Add to MLX weights
        mlx_weights[new_key] = mx.array(value).astype(dtype)
    
    # Verify critical layers are present
    critical_prefixes = [
        "encoder.conv1", "decoder.conv1", "conv1", "conv2",
        "encoder.head.layers.2", "decoder.head.layers.2"  # Updated for Sequential
    ]
    
    for prefix in critical_prefixes:
        found = any(k.startswith(prefix) for k in mlx_weights.keys())
        if not found:
            print(f"WARNING: No weights found for {prefix}")
    
    print(f"Converted {len(mlx_weights)} parameters")
    
    # Print a few example keys for verification
    print("\nExample converted keys:")
    for i, key in enumerate(sorted(mlx_weights.keys())[:10]):
        print(f"  {key}")
    
    # Save
    if output_path.endswith('.safetensors'):
        mx.save_safetensors(output_path, mlx_weights)
    else:
        mx.savez(output_path, **mlx_weights)
    
    print(f"\nSaved to {output_path}")
    print("\nAll converted keys:")
    for key in sorted(mlx_weights.keys()):
        print(f"  {key}: {mlx_weights[key].shape}")
    return mlx_weights