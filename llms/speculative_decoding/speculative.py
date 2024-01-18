import argparse
import time

import mlx.core as mx
from decoder import SpeculativeDecoder
from model import load_model


def main(args):
    mx.random.seed(args.seed)

    spec_decoder = SpeculativeDecoder(
        model=load_model(args.model_name),
        draft_model=load_model(args.draft_model_name),
        tokenizer=args.model_name,
        color=args.color,
        delta=args.delta,
        num_draft=args.num_draft,
    )

    tic = time.time()
    print(args.prompt)
    if args.regular_decode:
        spec_decoder.generate(args.prompt, max_tokens=args.max_tokens)
    else:
        stats = spec_decoder.speculative_decode(args.prompt, max_tokens=args.max_tokens)
        print("=" * 10)
        print(f"Accepted {stats['n_accepted']} / {stats['n_draft']}.")
        print(f"Decoding steps {stats['n_steps']}.")

    toc = time.time()
    print("=" * 10)
    print(f"Full generation time {toc - tic:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Llama weights to MLX")
    parser.add_argument(
        "--num-draft",
        type=int,
        default=5,
        help="Number of draft tokens to use per decoding step.",
    )
    parser.add_argument(
        "--model-name",
        help="Name of the model.",
        default="t5-large",
    )
    parser.add_argument(
        "--draft-model-name",
        help="Name of the draft model.",
        default="t5-small",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed.",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--prompt",
        default="Translate the following from English to French: Let's go to the store and buy some groceries including eggs, avocadoes, and bread.",
        help="The prompt processed by the model.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.1,
        help="Lenience for accepting the proposal tokens.",
    )
    parser.add_argument(
        "--color", type=bool, default=False, help="Color the accepted draft tokens"
    )
    parser.add_argument(
        "--regular-decode",
        action="store_true",
        help="Use regular decoding instead of speculative decoding.",
    )
    args = parser.parse_args()
    main(args)
