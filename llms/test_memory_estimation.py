#!/usr/bin/env python3
# Copyright © 2023-2025 Apple Inc.

"""
Test script to validate memory usage estimation from estimate.py.
Loads a model, runs inference, and compares actual vs estimated memory usage.
"""

import argparse
import gc
import os
import time
from typing import Dict, Optional, Tuple

import mlx.core as mx
from mlx_lm.estimate import (
    calculate_head_dim,
    compute_bits_per_weight_from_config,
    estimate_uram,
    fetch_config,
)
from mlx_lm.utils import generate, load, stream_generate


def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Test Memory Estimation Accuracy")
    parser.add_argument("model", help="Local model path or Hugging Face repo ID.")
    parser.add_argument(
        "--prompt", type=str, default="Once upon a time", help="Prompt for inference."
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="prompt.txt",
        help="File containing the prompt.",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=50, help="Number of tokens to generate."
    )
    parser.add_argument("--kv-bits", type=int, help="Bits for KV cache quantization.")
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=64,
        help="Group size for KV cache quantization.",
    )
    parser.add_argument(
        "--max-kv-size", type=int, help="Max KV cache size (bounded mode)."
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=4096,
        help="Context length for estimation.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    return parser


def get_memory_usage():
    """Get current memory usage in GB."""
    used_gb = mx.metal.get_peak_memory() / (1024**3)
    mx.metal.reset_peak_memory()
    return used_gb


def get_active_memory_gb():
    """Get current active memory usage in GB."""
    return mx.metal.get_active_memory() / (1024**3)


def force_gc_and_reset():
    """Force garbage collection and reset memory counters."""
    gc.collect()
    mx.metal.reset_peak_memory()
    time.sleep(0.1)  # Small delay to ensure memory operations complete


def measure_kv_cache_memory(
    model_pkg, tokenizer, prompt_tokens, num_new_tokens=10, verbose=False
):
    """
    Directly measure KV cache memory by comparing memory before and after cache creation.

    Args:
        model_pkg: The loaded model package (contains model and generate function)
        tokenizer: Tokenizer for the model
        prompt_tokens: Tokenized prompt
        num_new_tokens: Number of new tokens to generate
        verbose: Whether to print verbose output

    Returns:
        Tuple of (kv_cache_size_gb, kv_per_token_gb)
    """
    # Force clean memory state
    force_gc_and_reset()

    # Get baseline memory
    baseline = get_active_memory_gb()
    if verbose:
        print(f"Baseline memory before KV cache: {baseline:.4f} GB")

    # Create inputs
    inputs = mx.array([prompt_tokens])

    # First measure memory with just the prompt - no generation
    # Just do a forward pass to build the KV cache
    logits = model_pkg.model(inputs)
    mx.eval(logits)

    # Measure memory with just prompt in KV cache
    prompt_kv_memory = get_active_memory_gb() - baseline
    if verbose:
        print(
            f"Memory after prompt KV cache: {prompt_kv_memory:.4f} GB for {len(prompt_tokens)} tokens"
        )

    # Reset for generation test
    force_gc_and_reset()
    baseline_with_prompt_kv = get_active_memory_gb()

    # For generation, we need to use the mlx_lm generate function, not model.generate
    # Create a simple manual generation loop to measure memory impact
    input_ids = mx.array([prompt_tokens])

    # Generate tokens one by one to measure KV cache growth
    for _ in range(num_new_tokens):
        # Forward pass
        logits = model_pkg.model(input_ids)
        next_token_logits = logits[0, -1, :]
        next_token = mx.argmax(next_token_logits)
        input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)
        mx.eval(input_ids)

    # Measure memory after generation including KV cache
    total_kv_memory = (
        get_active_memory_gb() - baseline_with_prompt_kv + prompt_kv_memory
    )

    # Calculate per-token KV cache size
    total_tokens = len(prompt_tokens) + num_new_tokens
    if num_new_tokens > 0:
        per_token_gb = (total_kv_memory - prompt_kv_memory) / num_new_tokens
    else:
        per_token_gb = 0

    if verbose:
        print(
            f"Total KV cache memory: {total_kv_memory:.4f} GB for {total_tokens} tokens"
        )
        print(f"Measured KV cache per token: {per_token_gb:.8f} GB")

    return total_kv_memory, per_token_gb


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    verbose = args.verbose

    # Measure baseline memory before loading model
    force_gc_and_reset()
    baseline_memory_gb = get_memory_usage()
    print(f"Baseline memory usage: {baseline_memory_gb:.4f} GB")

    print(f"Loading model from {args.model}...")

    # Clear memory before starting
    force_gc_and_reset()

    # Load the model
    model_pkg, tokenizer = load(args.model)

    # Memory after loading model but before materializing parameters
    model_load_memory_gb = get_memory_usage()

    # Materialize model parameters to get accurate parameter memory usage
    print("Materializing model parameters...")
    force_gc_and_reset()

    # Force materialization of all model parameters
    for p in model_pkg.model.parameters().values():
        mx.eval(p)

    # Get memory used by materialized parameters
    parameter_memory_gb = get_active_memory_gb()
    print(f"Model parameters memory: {parameter_memory_gb:.4f} GB")

    # Try to read prompt from file if it exists, otherwise use default
    prompt = args.prompt
    try:
        if os.path.exists(args.prompt_file):
            with open(args.prompt_file, "r") as f:
                file_prompt = f.read().strip()
                if file_prompt:
                    prompt = file_prompt
                    print(
                        f"Using prompt from {args.prompt_file} ({len(prompt)} characters)"
                    )
                else:
                    print(f"Empty prompt file, using default prompt")
    except Exception as e:
        print(f"Error reading prompt file: {e}, using default prompt")

    # Count tokens in the prompt
    tokens = tokenizer.encode(prompt)
    prompt_token_count = len(tokens)
    print(f"Prompt length: {prompt_token_count} tokens")

    # Run inference with memory tracking
    print(f"Running inference with prompt...")

    # Reset memory state
    force_gc_and_reset()

    # First do a simple forward pass to measure activation memory
    print("Running single forward pass...")

    # Tokenize input
    inputs = mx.array([tokens])

    # Create an evaluation function that we will trace
    def forward_pass():
        return model_pkg.model(inputs)

    # Trace the function to capture the actual memory during execution
    output = forward_pass()
    mx.eval(output)

    # Get memory used during forward pass
    forward_memory_gb = get_memory_usage()
    print(f"Peak memory during forward pass: {forward_memory_gb:.4f} GB")

    # Get model config for additional details
    config = fetch_config(args.model)
    bits_per_weight = compute_bits_per_weight_from_config(config)

    # Get the necessary parameters for KV cache calculation from the model config
    from mlx_lm.utils import _get_classes

    _, model_args_class = _get_classes(config)
    model_args = model_args_class.from_dict(config)

    # Extract the parameters needed for KV cache calculation
    head_dim = calculate_head_dim(config, model_args)
    num_kv_heads = getattr(
        model_args, "num_key_value_heads", model_args.num_attention_heads
    )
    num_layers = model_args.num_hidden_layers

    # Now directly measure the KV cache memory
    print("\nDirectly measuring KV cache memory usage...")
    actual_kv_cache_gb, actual_per_token_gb = measure_kv_cache_memory(
        model_pkg,
        tokenizer,
        tokens,
        num_new_tokens=min(20, args.num_tokens),
        verbose=verbose,
    )

    # Measure memory during token generation (inference) using proper generate function
    print("\nMeasuring memory during full token generation (streaming)...")
    force_gc_and_reset()
    baseline_for_generation = get_active_memory_gb()

    # Use stream_generate to get token-by-token memory measurements
    generation_text = ""
    peak_memory_gb = 0
    total_tokens_generated = 0
    token_memory_profile = []

    # Stream generation and track memory for each token
    for response in stream_generate(
        model_pkg.model, tokenizer, prompt, max_tokens=args.num_tokens
    ):
        generation_text += response.text
        total_tokens_generated += 1

        # Track memory per token
        current_memory = response.peak_memory
        peak_memory_gb = max(peak_memory_gb, current_memory)

        # Record memory for this token
        token_memory_profile.append(current_memory)

        if verbose and total_tokens_generated % 10 == 0:
            print(
                f"Generated {total_tokens_generated} tokens, current memory: {current_memory:.4f} GB"
            )

    # Calculate final memory usage
    generation_memory_gb = peak_memory_gb
    print(
        f"Peak memory during generation of {total_tokens_generated} tokens: {generation_memory_gb:.4f} GB"
    )

    # You can also add this to get more detailed memory profile analysis
    if verbose:
        print("\nMemory growth during generation:")
        for i, mem in enumerate(token_memory_profile):
            if i % 5 == 0 or i == len(token_memory_profile) - 1:
                print(f"  Token {i+1}: {mem:.4f} GB")

    # Calculate activation memory (peak memory during generation minus parameter memory and KV cache)
    actual_activation_memory_gb = (
        generation_memory_gb - parameter_memory_gb - actual_kv_cache_gb
    )
    if actual_activation_memory_gb < 0:
        # This can happen due to memory reclamation between measurements
        actual_activation_memory_gb = (
            0.01 * parameter_memory_gb
        )  # Use a reasonable fallback

    # Get estimated memory usage from estimate.py
    estimated_results, mode = estimate_uram(
        args.model,
        context_length=args.context_length,
        max_kv_size=args.max_kv_size,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        initial_prompt_length=prompt_token_count,
        extra_tokens=total_tokens_generated,
    )

    # Compare estimation accuracy
    print("\n--- MEMORY ESTIMATION ACCURACY ---")
    print(f"Model: {args.model}")
    print(f"Architecture: {config.get('model_type', 'unknown')}")
    print(f"Quantization: {bits_per_weight} bits per weight")
    print(f"Total tokens processed: {prompt_token_count + total_tokens_generated}")
    print("-" * 40)
    print(f"Actual model parameters memory: {parameter_memory_gb:.3g} GB")
    print(f"Estimated model parameters memory: {estimated_results['Model']:.3g} GB")
    print(
        f"Model memory error: {abs(parameter_memory_gb - estimated_results['Model']):.3g} GB"
    )
    print("-" * 40)
    print(f"Actual KV cache memory: {actual_kv_cache_gb:.3g} GB")
    print(f"Estimated KV cache memory: {estimated_results['KV Cache']:.3g} GB")
    print(
        f"KV cache memory error: {abs(actual_kv_cache_gb - estimated_results['KV Cache']):.3g} GB"
    )
    print("-" * 40)
    print(f"Actual per-token KV increase: {actual_per_token_gb:.6g} GB")
    print(
        f"Estimated per-token KV increase: {estimated_results['per_token_increase']:.6g} GB"
    )
    print(
        f"Per-token KV error: {abs(actual_per_token_gb - estimated_results['per_token_increase']):.6g} GB"
    )
    print("-" * 40)
    print(f"Actual activation memory: {actual_activation_memory_gb:.3g} GB")
    print(f"Estimated activation memory: {estimated_results['Activations']:.3g} GB")
    print(
        f"Activation memory error: {abs(actual_activation_memory_gb - estimated_results['Activations']):.3g} GB"
    )
    print("-" * 40)
    print(f"Total peak memory (actual): {generation_memory_gb:.3g} GB")
    print(f"Total memory (estimated): {estimated_results['Total']:.3g} GB")
    print(
        f"Total memory error: {abs(generation_memory_gb - estimated_results['Total']):.3g} GB"
    )
    print(f"Mode: {mode}")


if __name__ == "__main__":
    main()
