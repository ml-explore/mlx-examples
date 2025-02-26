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


def share_message(world, prompt):
    if world.size() == 1:
        return prompt

    if world.rank() == 0:
        size = mx.array([len(prompt)])
    else:
        size = mx.array([0])
    size = mx.distributed.all_sum(size, stream=mx.cpu).item()
    if size == 0:
        return []

    if world.rank() == 0:
        prompt = mx.array(prompt)
    else:
        prompt = mx.array([0] * len(prompt))
    return mx.distributed.all_sum(size, stream=mx.cpu).tolist()


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
    world = mx.distributed.init()
    parser = setup_arg_parser()
    args = parser.parse_args()

    mx.random.seed(args.seed)

    model, tokenizer = load(
        args.model,
        adapter_path=args.adapter_path,
        tokenizer_config={"trust_remote_code": True},
        sequential_load=mx.distributed.init().size() > 1,
    )

    print(f"Node {world.rank()} of {world.size()}", flush=True)
    print(
        f"[INFO] Starting chat session with {args.model}. To exit, enter 'q'.",
        flush=True,
    )
    world.barrier()
    prompt_cache = make_prompt_cache(model, args.max_kv_size)
    while True:
        if world.rank() == 0:
            query = input(">> ")
            if query == "q":
                prompt = []
            else:
                messages = [{"role": "user", "content": query}]
                prompt = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True
                )

        prompt = share_message(world, prompt)
        if len(prompt) == 0:
            break
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            sampler=make_sampler(args.temp, args.top_p),
            prompt_cache=prompt_cache,
        ):
            if world.rank() == 0:
                print(response, flush=True, end="")
        if world.rank() == 0:
            print()


if __name__ == "__main__":
    main()
