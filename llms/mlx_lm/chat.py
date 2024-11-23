# Copyright © 2023-2024 Apple Inc.

import argparse
import json

import mlx.core as mx

from .models.cache import make_prompt_cache
from .sample_utils import make_sampler
from .utils import load, stream_generate

DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 0
DEFAULT_MAX_TOKENS = 256
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
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
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
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
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
            args.max_tokens,
            sampler=make_sampler(args.temp, args.top_p),
            prompt_cache=prompt_cache,
        ):
            print(response.text, flush=True, end="")
        print()


if __name__ == "__main__":
    main()
