# Copyright © 2023-2024 Apple Inc.

import argparse
import json

import mlx.core as mx

from .models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache
from .utils import load, stream_generate

DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 0
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="Chat with an LLM")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the local model directory or Hugging Face repo.",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--max-tokens-per-sec",
        type=int,
        help="Maximum tokens to generate per second.",
        default=None,
    )
    parser.add_argument(
        "--max-tokens-per-sec",
        type=int,
        default=None,
        help="Maximum tokens to generate per second",
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="PRNG seed")
    parser.add_argument(
        "--max-kv-size",
        type=int,
        help="Set the maximum key-value cache size",
        default=None,
    )
    return parser


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    mx.random.seed(args.seed)

    model, tokenizer = load(
        args.model,
        adapter_path=args.adapter_path,
        tokenizer_config={"trust_remote_code": True},
    )

    print(f"[INFO] Starting chat session with {args.model}. To exit, enter 'q'.")
    prompt_cache = make_prompt_cache(model, args.max_kv_size)
    while True:
        query = input(">> ")
        if query == "q":
            break
        messages = [{"role": "user", "content": query}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            temp=args.temp,
            top_p=args.top_p,
            max_tokens_per_sec=args.max_tokens_per_sec,
            prompt_cache=prompt_cache,
            max_tokens=4096 # Ensure this is set to a reasonable limit
        ):
            print(response, flush=True, end="")
        print()


if __name__ == "__main__":
    main()
