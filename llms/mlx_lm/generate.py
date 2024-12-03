# Copyright © 2023-2024 Apple Inc.

import argparse
import json
import sys

import mlx.core as mx

from .models.cache import QuantizedKVCache, load_prompt_cache
from .sample_utils import make_sampler
from .utils import generate, load

DEFAULT_PROMPT = "hello"
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MIN_P = 0.0
DEFAULT_MIN_TOKENS_TO_KEEP = 1
DEFAULT_SEED = 0
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_QUANTIZED_KV_START = 5000


def str2bool(string):
    return string.lower() not in ["false", "f"]


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="LLM inference script")
    parser.add_argument(
        "--model",
        type=str,
        help=(
            "The path to the local model directory or Hugging Face repo. "
            f"If no model is specified, then {DEFAULT_MODEL} is used."
        ),
        default=None,
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--eos-token",
        type=str,
        default=None,
        help="End of sequence token for tokenizer",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt to be used for the chat template",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default=DEFAULT_PROMPT,
        help="Message to be processed by the model ('-' reads from stdin)",
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
    parser.add_argument(
        "--min-p", type=float, default=DEFAULT_MIN_P, help="Sampling min-p"
    )
    parser.add_argument(
        "--min-tokens-to-keep",
        type=int,
        default=DEFAULT_MIN_TOKENS_TO_KEEP,
        help="Minimum tokens to keep for min-p sampling.",
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
        "--verbose",
        type=str2bool,
        default=True,
        help="Log verbose output when 'True' or 'T' or only print the response when 'False' or 'F'",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        help="Set the maximum key-value cache size",
        default=None,
    )
    parser.add_argument(
        "--prompt-cache-file",
        type=str,
        default=None,
        help="A file containing saved KV caches to avoid recomputing them",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        help="Number of bits for KV cache quantization. "
        "Defaults to no quantization.",
        default=None,
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        help="Group size for KV cache quantization.",
        default=64,
    )
    parser.add_argument(
        "--quantized-kv-start",
        help="When --kv-bits is set, start quantizing the KV cache "
        "from this step onwards.",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
    )
    return parser


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    mx.random.seed(args.seed)

    # Load the prompt cache and metadata if a cache file is provided
    using_cache = args.prompt_cache_file is not None
    if using_cache:
        prompt_cache, metadata = load_prompt_cache(
            args.prompt_cache_file,
            return_metadata=True,
        )
        if isinstance(prompt_cache[0], QuantizedKVCache):
            if args.kv_bits is not None and args.kv_bits != prompt_cache[0].bits:
                raise ValueError(
                    "--kv-bits does not match the kv cache loaded from --prompt-cache-file."
                )
            if args.kv_group_size != prompt_cache[0].group_size:
                raise ValueError(
                    "--kv-group-size does not match the kv cache loaded from --prompt-cache-file."
                )

    # Building tokenizer_config
    tokenizer_config = (
        {} if not using_cache else json.loads(metadata["tokenizer_config"])
    )
    tokenizer_config["trust_remote_code"] = True
    if args.eos_token is not None:
        tokenizer_config["eos_token"] = args.eos_token

    model_path = args.model
    if using_cache:
        if model_path is None:
            model_path = metadata["model"]
        elif model_path != metadata["model"]:
            raise ValueError(
                f"Providing a different model ({model_path}) than that "
                f"used to create the prompt cache ({metadata['model']}) "
                "is an error."
            )
    model_path = model_path or DEFAULT_MODEL

    model, tokenizer = load(
        model_path,
        adapter_path=args.adapter_path,
        tokenizer_config=tokenizer_config,
    )

    if args.use_default_chat_template:
        if tokenizer.chat_template is None:
            tokenizer.chat_template = tokenizer.default_chat_template
    elif using_cache:
        tokenizer.chat_template = metadata["chat_template"]

    if not args.ignore_chat_template and (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    ):
        if args.system_prompt is not None:
            messages = [{"role": "system", "content": args.system_prompt}]
        else:
            messages = []
        messages.append(
            {
                "role": "user",
                "content": sys.stdin.read() if args.prompt == "-" else args.prompt,
            }
        )
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Treat the prompt as a suffix assuming that the prefix is in the
        # stored kv cache.
        if using_cache:
            messages[-1]["content"] = "<query>"
            test_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt = prompt[test_prompt.index("<query>") :]
    else:
        prompt = args.prompt

    sampler = make_sampler(args.temp, args.top_p, args.min_p, args.min_tokens_to_keep)
    response = generate(
        model,
        tokenizer,
        prompt,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        sampler=sampler,
        max_kv_size=args.max_kv_size,
        prompt_cache=prompt_cache if using_cache else None,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
    )
    if not args.verbose:
        print(response)


if __name__ == "__main__":
    main()
