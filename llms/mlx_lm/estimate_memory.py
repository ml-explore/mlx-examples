# Copyright © 2023-2025 Apple Inc.

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

from huggingface_hub import hf_hub_download, try_to_load_from_cache
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.utils import _get_classes


def fetch_metadata(model_path: str) -> Tuple[Dict, Optional[int]]:
    """Fetch config.json and optionally model.safetensors.index.json for weights size."""
    config = fetch_config(model_path)
    model_weight_size = None
    if not os.path.isdir(model_path):
        try:
            index_path = hf_hub_download(
                repo_id=model_path, filename="model.safetensors.index.json"
            )
            with open(index_path, "r") as f:
                index = json.load(f)
                model_weight_size = index.get("metadata", {}).get("total_size")
        except:
            pass
    return config, model_weight_size


def fetch_config(model_path: str) -> Dict:
    """Fetch or load config.json without downloading the full model, checking cache first."""
    if os.path.isdir(model_path):
        config_path = Path(model_path) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_path}")
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        cached_path = try_to_load_from_cache(
            repo_id=model_path, filename="config.json", repo_type="model"
        )
        if cached_path and os.path.exists(cached_path):
            with open(cached_path, "r") as f:
                return json.load(f)
        try:
            config_path = hf_hub_download(repo_id=model_path, filename="config.json")
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to fetch config.json from {model_path}: {str(e)}")


def compute_bits_per_weight_from_config(config: Dict) -> float:
    """Infer bits-per-weight from config, defaulting to 16 (FP16) if unquantized."""
    quantization = config.get("quantization", {})
    bits = quantization.get("bits")
    return float(bits) if bits is not None else 16.0


def calc_embedding_params(vocab_size: int, hidden_size: int) -> int:
    """Calculate parameters for the embedding layer."""
    return vocab_size * hidden_size


def calc_attention_params(args, hidden_size: int, num_attention_heads: int) -> int:
    """Calculate parameters for one attention layer, handling standard and LoRA variants.

    This function supports both standard multi-head attention (e.g., Mixtral, OLMoE) and
    LoRA-like attention (e.g., DeepSeek V3) by checking for q_lora_rank, allowing flexibility
    over the older hardcoded approach that assumed uniform QKV dimensions.
    """
    num_kv_heads = getattr(args, "num_key_value_heads", num_attention_heads)
    head_dim = getattr(args, "head_dim", None)
    if head_dim is None:
        head_dim = hidden_size // num_attention_heads
    has_bias = getattr(args, "attention_bias", False)

    # Standard attention (Q, K, V, O)
    if not hasattr(args, "q_lora_rank") or not args.q_lora_rank:
        q_params = hidden_size * (num_attention_heads * head_dim)
        k_params = hidden_size * (num_kv_heads * head_dim)
        v_params = k_params
        o_params = (num_attention_heads * head_dim) * hidden_size
    # LoRA-like attention (e.g., DeepSeek V3)
    else:
        q_head_dim = getattr(args, "qk_nope_head_dim", 0) + getattr(
            args, "qk_rope_head_dim", head_dim
        )
        v_head_dim = getattr(args, "v_head_dim", head_dim)
        qk_rope_dim = getattr(args, "qk_rope_head_dim", head_dim)
        q_params = hidden_size * args.q_lora_rank + args.q_lora_rank * (
            num_attention_heads * q_head_dim
        )
        k_params = hidden_size * (args.kv_lora_rank + qk_rope_dim)
        v_params = args.kv_lora_rank * (
            num_attention_heads * (q_head_dim - qk_rope_dim + v_head_dim)
        )
        o_params = (num_attention_heads * v_head_dim) * hidden_size

    total = q_params + k_params + v_params + o_params
    if has_bias:
        total += (
            num_attention_heads * head_dim + num_kv_heads * head_dim * 2 + hidden_size
        )

    return total


def calc_ffn_or_moe_params(
    args, hidden_size: int, intermediate_size: int, layer_idx: int
) -> int:
    """Calculate parameters for FFN or MoE layer, switching based on config.

    Unlike the previous hardcoded FFN-only calculation, this function dynamically handles
    Mixture of Experts (MoE) models like Mixtral, OLMoE, and DeepSeek V3 by detecting
    expert-related fields (num_experts, n_routed_experts) and adjusting for dense vs.
    MoE layers, supporting varied intermediate sizes and shared experts.
    """
    num_experts = max(
        getattr(args, "num_local_experts", 0),
        getattr(args, "num_experts", 0),
        getattr(args, "n_routed_experts", 0),
    )
    moe_intermediate_size = getattr(args, "moe_intermediate_size", intermediate_size)
    dense_up_to = (
        getattr(args, "first_k_dense_replace", 0)
        if num_experts
        else args.num_hidden_layers
    )
    has_bias = getattr(args, "mlp_bias", False)
    shared_experts = getattr(args, "n_shared_experts", 0)

    if num_experts and layer_idx >= dense_up_to:
        # MoE: gate + expert FFNs
        gate_params = hidden_size * num_experts
        expert_params = (
            num_experts * hidden_size * moe_intermediate_size * 3
        )  # gate_proj, up_proj, down_proj
        shared_params = (
            shared_experts * hidden_size * moe_intermediate_size * 3
            if shared_experts
            else 0
        )
        return gate_params + expert_params + shared_params
    else:
        # Dense FFN
        ffn_params = (
            hidden_size * intermediate_size * 2 + intermediate_size * hidden_size
        )
        if has_bias:
            ffn_params += intermediate_size * 2 + hidden_size
        return ffn_params


def calc_norm_params(args, hidden_size: int, num_attention_heads: int) -> int:
    """Calculate normalization parameters, adjusting for extra norms in complex models.

    This extends the old approach (fixed 2 norms per layer) by adding heuristic support
    for extra normalization layers (e.g., OLMoE's q_norm, k_norm) in MoE or LoRA models,
    improving accuracy over the simpler assumption of uniform RMSNorm usage.
    """
    num_kv_heads = getattr(args, "num_key_value_heads", num_attention_heads)
    head_dim = getattr(args, "head_dim", None)
    if head_dim is None:
        head_dim = hidden_size // num_attention_heads

    # Base: input + post-attention RMSNorm
    total = hidden_size * 2

    # Heuristic: extra norms for MoE or complex attention
    has_experts = any(
        getattr(args, attr, 0) > 0
        for attr in ["num_local_experts", "num_experts", "n_routed_experts"]
    )
    if has_experts or hasattr(args, "q_lora_rank"):
        total += (num_attention_heads * head_dim) + (num_kv_heads * head_dim)

    return total


def calculate_num_parameters(
    config: Dict, model_args_class: Optional[Type["BaseModelArgs"]] = None
) -> int:
    """Calculate the total number of parameters in a model based on its config.

    By splitting into separate functions, we now support diverse
    architectures while maintaining readability and avoiding model-specific hardcoding.
    """
    # Use the imported _get_classes function to get the ModelArgs class
    if model_args_class is None:
        _, model_args_class = _get_classes(config)

    args = model_args_class.from_dict(config)

    # Validate required fields
    required = [
        "hidden_size",
        "num_hidden_layers",
        "vocab_size",
        "num_attention_heads",
        "intermediate_size",
    ]
    missing = [field for field in required if getattr(args, field, None) is None]
    if missing:
        raise ValueError(f"Config missing required fields: {missing}")

    # Core config
    hidden_size = args.hidden_size
    num_layers = args.num_hidden_layers
    vocab_size = args.vocab_size
    num_attention_heads = args.num_attention_heads
    intermediate_size = args.intermediate_size

    # Total calculation
    total_params = calc_embedding_params(vocab_size, hidden_size)
    for layer in range(num_layers):
        total_params += calc_attention_params(args, hidden_size, num_attention_heads)
        total_params += calc_ffn_or_moe_params(
            args, hidden_size, intermediate_size, layer
        )
        total_params += calc_norm_params(args, hidden_size, num_attention_heads)
    total_params += hidden_size  # Final norm
    if not getattr(args, "tie_word_embeddings", True):
        total_params += hidden_size * vocab_size  # LM head

    return total_params


def calculate_head_dim(config: Dict, args: BaseModelArgs) -> int:
    """Infer head dimension dynamically from config or args."""
    head_dim = getattr(args, "head_dim", None)
    if head_dim is None:
        if "hidden_size" not in config or "num_attention_heads" not in config:
            raise ValueError(
                "Cannot compute head_dim: missing hidden_size or num_attention_heads"
            )
        head_dim = config["hidden_size"] // config["num_attention_heads"]
    return head_dim


def estimate_mem(
    model_path: str,
    context_length: int = 4096,
    max_kv_size: Optional[int] = None,
    kv_bits: Optional[int] = None,
    kv_group_size: Optional[int] = None,
    tokens_to_generate: int = 0,
) -> Tuple[Dict[str, float], str]:
    """
    Estimate the memory usage of a model.

    Args:
        model_path: Path to the model.
        context_length: Context length of the model (prompt length in unbounded mode).
        max_kv_size: Maximum size of the KV cache (for bounded mode).
        kv_bits: Number of bits to use for quantized KV cache.
        kv_group_size: Group size to use for quantized KV cache.
        tokens_to_generate: Number of tokens to generate beyond the prompt.

    Returns:
        A tuple of (results, mode) where results is a dictionary of memory usage
        and mode is a string indicating the mode of the KV cache.
    """
    config, model_weight_size = fetch_metadata(model_path)
    bits_per_weight = compute_bits_per_weight_from_config(config)

    # Determine the model class
    _, model_args_class = _get_classes(config)
    args = model_args_class.from_dict(config)

    # Calculate the number of parameters
    num_parameters = calculate_num_parameters(config, model_args_class)

    # Extract model architecture parameters needed for memory calculations
    num_layers = args.num_hidden_layers
    num_kv_heads = getattr(args, "num_key_value_heads", args.num_attention_heads)
    head_dim = calculate_head_dim(config, args)

    # Default to fp16 (2 bytes per element) for KV cache unless quantized
    bytes_per_element = 2

    # If kv_bits and kv_group_size are not provided, try to read from config
    if kv_bits is None or kv_group_size is None:
        # Try to get quantization settings from config
        quantization = config.get("quantization", {})
        quantization_config = config.get("quantization_config", {})

        # Use the first available quantization config
        quant_info = quantization or quantization_config

        if quant_info:
            kv_bits = kv_bits or quant_info.get("bits")
            kv_group_size = kv_group_size or quant_info.get("group_size")

    # Calculate the model weight memory usage
    bytes_per_parameter = bits_per_weight / 8
    if model_weight_size:
        # Use the size from safetensors index if available
        model_size_gb = model_weight_size / (1024**3)
    else:
        # Calculate from parameter count
        model_size_gb = (num_parameters * bytes_per_parameter) / (1024**3)

    # Estimate tokenizer size
    vocab_size = config.get("vocab_size", args.vocab_size)
    fixed_overhead_bytes = 25 * 1024 * 1024
    avg_token_size_bytes = 650
    tokenizer_size_bytes = (vocab_size * avg_token_size_bytes) + fixed_overhead_bytes
    tokenizer_size_gb = tokenizer_size_bytes / (1024**3)

    # Determine the mode
    mode = "Bounded" if max_kv_size else "Unbounded"

    # KV length is fixed to max_kv_size in bounded mode, or context_length in unbounded mode
    kv_length = max_kv_size if mode == "Bounded" else context_length

    # Default step size from cache.py is 256
    step_size = 256
    kv_length_padded = ((kv_length + step_size - 1) // step_size) * step_size

    # Calculate KV cache size based on whether quantization is used
    if kv_bits and kv_group_size:
        # Quantized cache calculations
        groups_per_head_dim = (head_dim + kv_group_size - 1) // kv_group_size
        elements_per_int = 8 * 4 // kv_bits

        data_size = (
            num_kv_heads * kv_length_padded * (head_dim // elements_per_int) * 4
        ) / (1024**3)
        quant_overhead = (
            num_kv_heads * kv_length_padded * groups_per_head_dim * 2 * 2
        ) / (1024**3)
        per_layer_kv_size = 2 * (data_size + quant_overhead)

        elements_per_token = (head_dim // elements_per_int) * 4
        scales_zeros_per_token = groups_per_head_dim * 2 * 2
        per_token_bytes = (
            2 * num_kv_heads * (elements_per_token + scales_zeros_per_token)
        )
        per_token_increase = (per_token_bytes * num_layers) / (1024**3)
    else:
        # Standard fp16 cache
        per_layer_kv_size = (
            2 * num_kv_heads * kv_length_padded * head_dim * bytes_per_element
        ) / (1024**3)
        per_token_increase = (
            2 * num_kv_heads * head_dim * bytes_per_element * num_layers
        ) / (1024**3)

    total_kv_cache_size = num_layers * per_layer_kv_size

    # Add the memory for generated tokens if specified
    if tokens_to_generate > 0:
        total_kv_cache_size += tokens_to_generate * per_token_increase

    # For inference in MLX, estimate activation memory
    activation_size_gb = 0.03 * model_size_gb

    overhead_gb = tokenizer_size_gb + activation_size_gb + (model_size_gb * 0.05)

    # Total memory usage
    total_memory_gb = model_size_gb + total_kv_cache_size + overhead_gb

    results = {
        "Weight": model_size_gb,
        "KV Cache": total_kv_cache_size,
        "Overhead": overhead_gb,
        "Total": total_memory_gb,
        "per_token_increase": per_token_increase,
    }

    return results, mode


def setup_arg_parser():
    parser = argparse.ArgumentParser(description="MLX Model URAM Estimation Tool")
    parser.add_argument("model", help="Local model path or Hugging Face repo ID.")
    parser.add_argument(
        "--context-length",
        type=int,
        default=4096,
        help="Context length of the model (prompt length in unbounded mode).",
    )
    parser.add_argument(
        "--max-kv-size", type=int, help="Max KV cache size (enables bounded mode)."
    )
    parser.add_argument("--kv-bits", type=int, help="Bits for KV cache quantization.")
    parser.add_argument(
        "--kv-group-size", type=int, help="Group size for KV cache quantization."
    )
    parser.add_argument(
        "--tokens-to-generate",
        type=int,
        default=0,
        help="Number of tokens to generate beyond the prompt.",
    )
    return parser


def print_table(
    results: Dict[str, float], mode: str, tokens_to_generate: int = 0
) -> None:
    """
    Print a memory usage table in a formatted way.

    Args:
        results: Dictionary containing memory usage data (in GB unless specified).
        mode: Either "Bounded" or "Unbounded" to describe the KV Cache type.
        tokens_to_generate: Number of tokens generated (optional, defaults to 0).
    """
    # Construct the title dynamically
    title = f"*Memory Usage Estimate ({mode} KV Cache"
    if tokens_to_generate > 0:
        title += f" after generating {tokens_to_generate:,} tokens"
    title += "):*"

    # Define table formatting constants
    LINE_WIDTH = 34
    ITEM_COL_WIDTH = 17
    MEMORY_COL_WIDTH = 12

    # Print header
    print(title)
    print("-" * LINE_WIDTH)
    print(f"| {'Item':<{ITEM_COL_WIDTH}} | {'Memory':<{MEMORY_COL_WIDTH}} |")
    print("-" * LINE_WIDTH)

    # Define display order and handle missing keys gracefully
    display_order = ["Weight", "KV Cache", "Overhead", "Total"]
    for key in display_order:
        value = results.get(key, 0.0)  # Default to 0.0 if key is missing
        memory_str = f"{value:.2f} GB"

        # Print row (extra spaces for alignment)
        if key == "Total":
            print("-" * LINE_WIDTH)
        print(f"| {key:<{ITEM_COL_WIDTH}} | {memory_str:>{MEMORY_COL_WIDTH}} |")

    print("-" * LINE_WIDTH)

    # Add footer for Unbounded mode
    if mode == "Unbounded" and "per_token_increase" in results:
        per_token_gb = results["per_token_increase"]
        if per_token_gb > 0:  # Avoid division by zero
            tokens_per_gb = math.floor(1 / per_token_gb)
            print(f"Additional tokens per 1GB increase: {tokens_per_gb:,}")
        else:
            print("Note: Per-token increase is zero or invalid.")


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    results, mode = estimate_mem(
        args.model,
        args.context_length,
        args.max_kv_size,
        args.kv_bits,
        args.kv_group_size,
        args.tokens_to_generate,
    )

    print_table(results, mode, args.tokens_to_generate)


if __name__ == "__main__":
    main()
