import argparse

import mlx.core as mx
from utils import generate, load

DEFAULT_MODEL_PATH = "mlx_model"
DEFAULT_PROMPT = "hello"
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMP = 0.6
DEFAULT_SEED = 0


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="LLM inference script")
    parser.add_argument(
        "--hf-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the mlx model weights, tokenizer, and config",
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
    return parser


def main(args):
    mx.random.seed(args.seed)
    model, tokenizer = load(args.hf_path)
    prompt = tokenizer.encode(args.prompt)
    prompt = mx.array(prompt)

    tokens = []
    for token, _ in zip(generate(prompt, model, args.temp), range(args.max_tokens)):
        if token == tokenizer.eos_token_id:
            break
        tokens.append(token.item())

    output = tokenizer.decode(tokens)
    print(output, flush=True)


if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args()
    main(args)
