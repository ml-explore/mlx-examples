import os
import torch
from safetensors.torch import save_file
from pathlib import Path
import json

from wan.modules.t5 import T5Model


def convert_pickle_to_safetensors(
    pickle_path: str, 
    safetensors_path: str,
    model_class=None,
    model_kwargs=None,
    load_method: str = "weights_only"  # Changed default to weights_only
):
    """Convert PyTorch pickle file to safetensors format."""
    
    print(f"Loading PyTorch weights from: {pickle_path}")
    
    # Try multiple loading methods in order of preference
    methods_to_try = [load_method, "weights_only", "state_dict", "full_model"]
    
    for method in methods_to_try:
        try:
            if method == "weights_only":
                state_dict = torch.load(pickle_path, map_location='cpu', weights_only=True)
                
            elif method == "state_dict":
                checkpoint = torch.load(pickle_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
                    
            elif method == "full_model":
                model = torch.load(pickle_path, map_location='cpu')
                if hasattr(model, 'state_dict'):
                    state_dict = model.state_dict()
                else:
                    state_dict = model
                    
            print(f"✅ Successfully loaded with method: {method}")
            break
            
        except Exception as e:
            print(f"❌ Method {method} failed: {e}")
            continue
    else:
        raise RuntimeError(f"All loading methods failed for {pickle_path}")
    
    # Clean up the state dict
    state_dict = clean_state_dict(state_dict)
    
    print(f"Found {len(state_dict)} parameters")
    
    # Convert BF16 to FP32 if needed
    for key, tensor in state_dict.items():
        if tensor.dtype == torch.bfloat16:
            state_dict[key] = tensor.to(torch.float32)
            print(f"Converted {key} from bfloat16 to float32")
    
    # Save as safetensors
    print(f"Saving to safetensors: {safetensors_path}")
    os.makedirs(os.path.dirname(safetensors_path), exist_ok=True)
    save_file(state_dict, safetensors_path)
    
    print("✅ T5 conversion complete!")
    return state_dict


def clean_state_dict(state_dict):
    """
    Clean up state dict by removing unwanted prefixes or keys.
    """
    cleaned = {}
    
    for key, value in state_dict.items():
        # Remove common prefixes that might interfere
        clean_key = key
        
        if clean_key.startswith('module.'):
            clean_key = clean_key[7:]
            
        if clean_key.startswith('model.'):
            clean_key = clean_key[6:]
            
        cleaned[clean_key] = value
        
        if clean_key != key:
            print(f"Cleaned key: {key} -> {clean_key}")
    
    return cleaned


def load_with_your_torch_model(pickle_path: str, model_class, **model_kwargs):
    """
    Load pickle weights into your specific PyTorch T5 model implementation.
    
    Args:
        pickle_path: Path to pickle file
        model_class: Your T5Encoder class
        **model_kwargs: Arguments for your model constructor
    """
    
    print("Method 1: Loading into your PyTorch T5 model")
    
    # Initialize your model
    model = model_class(**model_kwargs)
    
    # Load checkpoint
    checkpoint = torch.load(pickle_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            # Assume the dict IS the state dict
            state_dict = checkpoint
    else:
        # Assume it's a model object
        state_dict = checkpoint.state_dict()
    
    # Clean and load
    state_dict = clean_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)  # Use strict=False to ignore missing keys
    
    return model, state_dict


def explore_pickle_file(pickle_path: str):
    """
    Explore the contents of a pickle file to understand its structure.
    """
    print(f"🔍 Exploring pickle file: {pickle_path}")
    
    try:
        # Try loading with weights_only first (safer)
        print("\n--- Trying weights_only=True ---")
        try:
            data = torch.load(pickle_path, map_location='cpu', weights_only=True)
            print(f"✅ Loaded with weights_only=True")
            print(f"Type: {type(data)}")
            
            if isinstance(data, dict):
                print(f"Dictionary with {len(data)} keys:")
                for i, key in enumerate(data.keys()):
                    print(f"  {key}: {type(data[key])}")
                    if hasattr(data[key], 'shape'):
                        print(f"    Shape: {data[key].shape}")
                    if i >= 9:  # Show first 10 keys
                        break
                        
        except Exception as e:
            print(f"❌ weights_only=True failed: {e}")
    
        # Try regular loading
        print("\n--- Trying regular loading ---")
        data = torch.load(pickle_path, map_location='cpu')
        print(f"✅ Loaded successfully")
        print(f"Type: {type(data)}")
        
        if hasattr(data, 'state_dict'):
            print("📋 Found state_dict method")
            state_dict = data.state_dict()
            print(f"State dict has {len(state_dict)} parameters")
            
        elif isinstance(data, dict):
            print(f"📋 Dictionary with keys: {list(data.keys())}")
            
            # Check for common checkpoint keys
            if 'state_dict' in data:
                print("Found 'state_dict' key")
                print(f"state_dict has {len(data['state_dict'])} parameters")
            elif 'model' in data:
                print("Found 'model' key")
                print(f"model has {len(data['model'])} parameters")
        
    except Exception as e:
        print(f"❌ Failed to load: {e}")


def full_conversion_pipeline(
    pickle_path: str,
    safetensors_path: str,
    torch_model_class=None,
    model_kwargs=None
):
    """
    Complete pipeline: pickle -> safetensors -> ready for MLX conversion
    """
    
    print("🚀 Starting full conversion pipeline")
    print("="*50)
    
    # Step 1: Explore the pickle file
    print("Step 1: Exploring pickle file structure")
    explore_pickle_file(pickle_path)
    
    # Step 2: Convert to safetensors
    print(f"\nStep 2: Converting to safetensors")
    
    # Try different loading methods
    for method in ["weights_only", "state_dict", "full_model"]:
        try:
            print(f"\nTrying load method: {method}")
            state_dict = convert_pickle_to_safetensors(
                pickle_path, 
                safetensors_path,
                model_class=torch_model_class,
                model_kwargs=model_kwargs,
                load_method=method
            )
            print(f"✅ Success with method: {method}")
            break
            
        except Exception as e:
            print(f"❌ Method {method} failed: {e}")
            continue
    else:
        print("❌ All methods failed!")
        return None
    
    print(f"\n🎉 Conversion complete!")
    print(f"Safetensors file saved to: {safetensors_path}")
    print(f"Ready for MLX conversion using the previous script!")
    
    return state_dict