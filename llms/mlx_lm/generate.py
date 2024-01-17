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
    model, tokenizer = load(args.model)
    print("=" * 10)
    print("Prompt:", args.prompt)
    prompt = tokenizer.encode(args.prompt)
    prompt = mx.array(prompt)
    tic = time.time()
    tokens = []
    skip = 0
    for token, n in zip(
        generate_step(prompt, model, args.temp), range(args.max_tokens)
    ):
        if token == tokenizer.eos_token_id:
            break
        if n == 0:
            prompt_time = time.time() - tic
            tic = time.time()
        tokens.append(token.item())
        s = tokenizer.decode(tokens)
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
