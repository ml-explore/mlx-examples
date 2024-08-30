# Copyright © 2023-2024 Apple Inc.

import argparse
import json

import mlx.core as mx

from .utils import generate, load

DEFAULT_PROMPT = "hello"
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMP = 0.6
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 0


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="LLM inference script")
    parser.add_argument(
        "--model",
        type=str,
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
        "--prompt", default=DEFAULT_PROMPT, help="Message to be processed by the model"
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="PRNG seed")
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
        "--colorize",
        action="store_true",
        help="Colorize output based on T[0] probability",
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
        help="Set the maximum key-value cache size",
        default=None,
    )
    parser.add_argument(
        "--kv-cache-file",
        type=str,
        default=None,
        help="A file containing saved KV caches to avoid recomputing them",
    )
    return parser


def colorprint(color, s):
    color_codes = {
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "white": 39,
    }
    ccode = color_codes.get(color, 30)
    print(f"\033[1m\033[{ccode}m{s}\033[0m", end="", flush=True)


def colorprint_by_t0(s, t0):
    if t0 > 0.95:
        color = "white"
    elif t0 > 0.70:
        color = "green"
    elif t0 > 0.30:
        color = "yellow"
    else:
        color = "red"
    colorprint(color, s)


def load_kv_cache_from_file(kv_cache_file):
    if kv_cache_file is None:
        return None, None

    kv_cache, metadata = mx.load(kv_cache_file, return_metadata=True)
    cache_per_layer = {}
    for k, x in kv_cache.items():
        layer, kv_type = k.split("_")
        if layer not in cache_per_layer:
            cache_per_layer[layer] = {}
        cache_per_layer[layer][kv_type] = x

    cache_history = [None] * len(cache_per_layer)
    for layer, c in cache_per_layer.items():
        cache_history[int(layer)] = (c["keys"], c["values"])
    return cache_history, metadata


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    mx.random.seed(args.seed)

    if args.cache_limit_gb is not None:
        mx.metal.set_cache_limit(args.cache_limit_gb * 1024 * 1024 * 1024)

    # Load the kv cache and metadata if a kv cache file is provided
    cache_history, metadata = load_kv_cache_from_file(args.kv_cache_file)

    # Building tokenizer_config
    tokenizer_config = (
        {} if cache_history is None else json.loads(metadata["tokenizer_config"])
    )
    if args.trust_remote_code:
        tokenizer_config["trust_remote_code"] = True
    if args.eos_token is not None:
        tokenizer_config["eos_token"] = args.eos_token

    # If no model path is provided then use the one in the kv cache history
    model_path = args.model
    if cache_history is not None and model_path is None:
        model_path = metadata["model"]

    model, tokenizer = load(
        model_path,
        adapter_path=args.adapter_path,
        tokenizer_config=tokenizer_config,
    )

    if args.use_default_chat_template:
        if tokenizer.chat_template is None:
            tokenizer.chat_template = tokenizer.default_chat_template
    elif cache_history is not None:
        tokenizer.chat_template = metadata["chat_template"]

    if not args.ignore_chat_template and (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    ):
        messages = [{"role": "user", "content": args.prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Treat the prompt as a suffix assuming that the prefix is in the
        # stored kv cache.
        if cache_history is not None:
            test_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": "<query>"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt = prompt[test_prompt.index("<query>") :]
    else:
        prompt = args.prompt

    formatter = colorprint_by_t0 if args.colorize else None

    # Determine the max kv size from the kv cache or passed arguments
    max_kv_size = args.max_kv_size
    if cache_history is not None:
        max_kv_size = metadata["max_kv_size"]
        max_kv_size = int(max_kv_size) if max_kv_size.isdigit() else None

    generate(
        model,
        tokenizer,
        prompt,
        args.max_tokens,
        verbose=True,
        formatter=formatter,
        temp=args.temp,
        top_p=args.top_p,
        max_kv_size=max_kv_size,
        cache_history=cache_history,
    )


if __name__ == "__main__":
    main()
