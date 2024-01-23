import argparse
import time

import mlx.core as mx

from .utils import generate_step, load

DEFAULT_MODEL_PATH = "mlx_model"
DEFAULT_PROMPT = "hello"
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMP = 0.6
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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="PRNG seed")
    parser.add_argument(
        "--ignore-chat-template",
        action="store_true",
        help="Use the raw prompt without the tokenizer's chat template.",
    )
    parser.add_argument(
        "--colorize",
        action='store_true',
        help="Colorize output based on T[0] probability",
    )
    return parser


def colorprint(color, s):
    color_codes = {
        'black': 30,
        'red': 31,
        'green': 32,
        'yellow': 33,
        'blue': 34,
        'magenta': 35,
        'cyan': 36,
        'white': 39,
    }
    ccode = color_codes.get(color, 30)
    print(f"\033[1m\033[{ccode}m{s}\033[0m", end="", flush=True)


def colorprint_by_t0(t0, s):
    if t0 > 0.95:
        color = 'white'
    elif t0 > 0.70:
        color = 'green'
    elif t0 > 0.30:
        color = 'yellow'
    else:
        color = 'red'
    colorprint(color,s)


def main(args):
    mx.random.seed(args.seed)

    # Building tokenizer_config
    tokenizer_config = {"trust_remote_code": True if args.trust_remote_code else None}
    if args.eos_token is not None:
        tokenizer_config["eos_token"] = args.eos_token

    model, tokenizer = load(args.model, tokenizer_config=tokenizer_config)

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

    print("=" * 10)
    print("Prompt:", prompt)
    prompt = tokenizer.encode(prompt)
    prompt = mx.array(prompt)
    tic = time.time()
    tokens = []
    skip = 0
    for token, n in zip(
        generate_step(prompt, model, args.temp, args.colorize), range(args.max_tokens)
    ):
        token, t0 = token
        if token == tokenizer.eos_token_id:
            break
        if n == 0:
            prompt_time = time.time() - tic
            tic = time.time()
        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        if args.colorize:
            colorprint_by_t0(t0,s[skip:])
        else:
            print(s[skip:], end="", flush=True)
        skip = len(s)
    print(tokenizer.decode(tokens)[skip:], flush=True)
    gen_time = time.time() - tic
    print("=" * 10)
    if len(tokens) == 0:
        print("No tokens generated for this prompt")
        return
    prompt_tps = prompt.size / prompt_time
    gen_tps = (len(tokens) - 1) / gen_time
    print(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
    print(f"Generation: {gen_tps:.3f} tokens-per-sec")


if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args()
    main(args)
