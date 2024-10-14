# Copyright © 2024 Apple Inc.

import argparse
import json
import sys
import time

import mlx.core as mx

from .models.cache import make_prompt_cache, save_prompt_cache
from .utils import load


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Cache the state of a prompt to be reused with mlx_lm.generate"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--eos-token",
        type=str,
        default=None,
        help="End of sequence token for tokenizer",
    )
    parser.add_argument(
        "--ignore-chat-template",
        action="store_true",
        help="Use the raw prompt without the tokenizer's chat template.",
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--cache-limit-gb",
        type=int,
        default=None,
        help="Set the MLX cache limit in GB",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Set the maximum key-value cache size",
    )
    parser.add_argument(
        "--prompt-cache-file",
        help="The file to save the prompt cache in",
        required=True,
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Message to be processed by the model ('-' reads from stdin)",
    )
    return parser


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    if args.cache_limit_gb is not None:
        mx.metal.set_cache_limit(args.cache_limit_gb * 1024 * 1024 * 1024)

    # Building tokenizer_config
    tokenizer_config = {"trust_remote_code": True if args.trust_remote_code else None}
    if args.eos_token is not None:
        tokenizer_config["eos_token"] = args.eos_token

    model, tokenizer = load(
        args.model,
        adapter_path=args.adapter_path,
        tokenizer_config=tokenizer_config,
    )

    args.prompt = sys.stdin.read() if args.prompt == "-" else args.prompt

    if args.use_default_chat_template:
        if tokenizer.chat_template is None:
            tokenizer.chat_template = tokenizer.default_chat_template

    if not args.ignore_chat_template and (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    ):
        messages = [{"role": "user", "content": args.prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Treat the prompt as a prefix assuming that the suffix will be
        # provided at generation time.
        test_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "<query>"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        n = len(test_prompt) - test_prompt.index("<query>") - len("<query>")
        prompt = prompt[:-n]
    else:
        prompt = args.prompt

    cache = make_prompt_cache(model, args.max_kv_size)
    y = mx.array(tokenizer.encode(prompt))

    # Process the prompt
    processed = 0
    step_size = 512
    start = time.time()
    max_msg_len = 0
    while y.size > 0:
        model(y[:step_size][None], cache=cache)
        mx.eval([c.state for c in cache])
        processed += min(y.size, step_size)
        y = y[step_size:]
        current = time.time()
        speed = processed / (current - start)
        msg = f"\rProcessed {processed:6d} tokens ({speed:6.2f} tok/s)"
        max_msg_len = max(max_msg_len, len(msg))
        print(msg + " " * (max_msg_len - len(msg)), end="", flush=True)
    print()
    print(f"Peak memory: {mx.metal.get_peak_memory() / 2**30:.3f} GB")

    print("Saving...")
    metadata = {}
    metadata["model"] = args.model
    metadata["chat_template"] = tokenizer.chat_template
    metadata["tokenizer_config"] = json.dumps(tokenizer_config)
    print(f"Peak memory: {mx.metal.get_peak_memory() / 2**30:.3f} GB")
    save_prompt_cache(args.prompt_cache_file, cache, metadata)


if __name__ == "__main__":
    main()
