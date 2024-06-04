# Copyright © 2023-2024 Apple Inc.

import argparse

import mlx.core as mx

from .utils import generate, load

DEFAULT_MODEL_PATH = "mlx_model"
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
        required=False,
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


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    mx.random.seed(args.seed)

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
    else:
        prompt = args.prompt

    formatter = colorprint_by_t0 if args.colorize else None

    generate(
        model,
        tokenizer,
        prompt,
        args.max_tokens,
        verbose=True,
        formatter=formatter,
        temp=args.temp,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
