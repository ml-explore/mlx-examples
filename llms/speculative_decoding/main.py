import argparse

import mlx.core as mx
from decoder import SpeculativeDecoder


def main(args):
    mx.random.seed(args.seed)

    spec_decoder = SpeculativeDecoder(
        # model="TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T",
        model="meta-llama/Llama-2-7b-hf",
        draft_model="TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T",
        delta=args.delta,
        num_draft=args.num_draft,
    )

    prompt = {"role": "user", "content": "Finish the monologue: To be, or not to be..."}

    # Do 1 regular generation to get warmed up (the first one is slow)
    # engine.generate(messages, max_tokens=1)
    # engine.generate(messages, max_tokens=1, draft=True)

    # Time regular generation
    spec_decoder.generate(prompt, max_tokens=125)

    # Time speculative decoding
    spec_decoder.speculative_decode(prompt, max_tokens=125)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Llama weights to MLX")
    parser.add_argument(
        "--num-draft",
        type=int,
        default=5,
        help="Number of draft tokens to use per decoding step.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.1,
        help="Lenience for accepting the proposal tokens.",
    )
    args = parser.parse_args()
    main(args)
