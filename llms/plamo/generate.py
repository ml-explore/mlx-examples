# Copyright © 2023 Apple Inc.

import argparse
import time

import mlx.core as mx
from sentencepiece import SentencePieceProcessor

from modeling_plamo import PlamoForCausalLM, load_model


def tic():
    return time.time()


def toc(msg, start):
    end = time.time()
    return f"[INFO] {msg}: {end - start:.3f} s"


def generate(
    model: PlamoForCausalLM,
    tokenizer: SentencePieceProcessor,
    prompt: str,
    max_tokens: int,
    write_every: int,
    temp: float = 0.0,
):
    # input("Press enter to start generation")
    print("------")
    print(prompt)
    x = mx.array([[tokenizer.bos_id()] + tokenizer.encode(prompt)])
    skip = 0
    prompt_processing = None
    tokens = []
    start = tic()

    for token in model.generate(x, temp):
        tokens.append(token)

        if len(tokens) == 1:
            # Actually perform the computation to measure the prompt processing time
            mx.eval(token)
            prompt_processing = toc("Prompt processing", start)

        if len(tokens) >= max_tokens:
            break

        elif (len(tokens) % write_every) == 0:
            # It is perfectly ok to eval things we have already eval-ed.
            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            print(s[skip:], end="", flush=True)
            skip = len(s)

    mx.eval(tokens)
    full_gen = toc("Full generation", start)
    s = tokenizer.decode([t.item() for t in tokens])
    print(s[skip:], flush=True)
    print("------")
    print(prompt_processing)
    print(full_gen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference script")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model",
        default="In the beginning the Universe was created.",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--write-every", type=int, default=1, help="After how many tokens to detokenize"
    )
    parser.add_argument(
        "--temp",
        help="The sampling temperature.",
        type=float,
        default=0.0,
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")

    args = parser.parse_args()
    mx.random.seed(args.seed)

    model, tokenizer = load_model(args.model)
    generate(
        model, tokenizer, args.prompt, args.max_tokens, args.write_every, args.temp
    )
