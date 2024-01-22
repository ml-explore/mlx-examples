import argparse

import mlx.core as mx

from .utils import generate, load

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
    parser.add_argument('-v', '--verbose', help='Increase output verbosity', action='store_true')
    return parser


def main(args):
    model, tokenizer = load(args.model)
    output = generate(model, tokenizer, args.prompt, args.temp, args.max_tokens, args.seed, args.verbose)
    if not args.verbose:
        print(output)


if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args()
    main(args)
