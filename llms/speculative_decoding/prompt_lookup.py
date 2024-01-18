import argparse
import time

import mlx.core as mx
from decoder import PromptLookupDecoder
from model import load_model


def main(args):
    mx.random.seed(args.seed)

    lookup_decoder = PromptLookupDecoder(
        model=load_model(args.model_name),
        tokenizer=args.model_name,
        n_draft=args.n_draft,
        ngram_max=args.ngram_max,
        ngram_min=args.ngram_min,
        temp=args.temp,
        seed=args.seed,
        color=args.color,
    )

    tic = time.time()
    print(args.prompt)

    stats = lookup_decoder.prompt_lookup(args.prompt, max_tokens=args.max_tokens)
    print("=" * 10)
    print(f"Accepted {stats['n_accepted']} / {stats['n_draft']}.")
    print(f"Decoding steps {stats['n_steps']}.")

    toc = time.time()
    print("=" * 10)
    print(f"Full generation time {toc - tic:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt Lookup Decoding")

    parser.add_argument(
        "--n-draft",
        type=int,
        default=10,
        help="Number of draft tokens to generate upon prompt lookup match",
    )
    parser.add_argument(
        "--model-name",
        help="Name of the model.",
        default="t5-base",
    )

    parser.add_argument(
        "--prompt",
        help="The prompt processed by the model.",
        default="Repeat the following sentence 5 times: 'The quick brown fox jumped over the fence.'",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=3,
        help="Maximum ngrams to match against input during prompt lookup",
    )
    parser.add_argument(
        "--ngram-min",
        type=int,
        default=1,
        help="Minimum ngrams to match against input during prompt lookup",
    )
    parser.add_argument(
        "--temp",
        help="The sampling temperature.",
        type=float,
        default=0.0,
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    parser.add_argument(
        "--color", type=bool, default=False, help="Color the accepted draft tokens"
    )

    args = parser.parse_args()

    main(args)
