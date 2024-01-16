# Copyright © 2023 Apple Inc.

import argparse
import time

import mlx.core as mx
from plamo import PlamoForCausalLM, load_model
from sentencepiece import SentencePieceProcessor


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
) -> str:
    # input("Press enter to start generation")
    print("------")
    x = mx.array([tokenizer.encode(prompt)], dtype=mx.int64)
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
    return s


# From: https://huggingface.co/pfnet/plamo-13b-instruct
def generate_prompt(messages: list) -> str:
    sep = "\n\n### "
    prompt = [
        "以下はタスクを説明する指示で、文脈を説明した入力とペアになっています。",
        "要求を適切に補完するよう応答を書いてください。",
    ]
    roles = {"instruction": "指示", "response": "応答", "input": "入力"}
    for msg in messages:
        prompt.append(sep + roles[msg["role"]] + ":\n" + msg["content"])
    prompt.append(sep + roles["response"] + ":\n")
    return "".join(prompt)


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
        default=128,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument("--write-every", type=int, default=1, help="After how many tokens to detokenize")
    parser.add_argument(
        "--temp",
        help="The sampling temperature.",
        type=float,
        default=0.7,
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    parser.add_argument("--instruct", "-i", action="store_true", help="Use the instruct prompt")

    args = parser.parse_args()
    mx.random.seed(args.seed)

    model, tokenizer = load_model(args.model)

    instruction_base = [
        {
            "role": "instruction",
            "content": args.prompt,
        },
    ]
    if args.instruct:
        prompt = generate_prompt(instruction_base)
    else:
        prompt = args.prompt

    answer = generate(model, tokenizer, prompt, args.max_tokens, args.write_every, args.temp)

    while True:
        new_input = input("Input more text to continue generation:")
        if new_input == "":
            break
        if args.instruct:
            instruction_base.append(
                {
                    "role": "input",
                    "content": new_input,
                },
            )
            prompt = generate_prompt(instruction_base)
        else:
            prompt = new_input

        answer = generate(model, tokenizer, prompt, args.max_tokens, args.write_every, args.temp)
