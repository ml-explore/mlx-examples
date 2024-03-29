# Copyright © 2023 Apple Inc.
# Edited by: Jaward Sesay (Jaykef) 2024-26-04
# File: rag_vdb.py - retrieves data from vdb used in queryiing the base model.

import argparse
import time
from vdb import VectorDB
import mlx.core as mx
import gguf

TEMPLATE = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

Question: what is the meaning of life?
Answer: The meaning of life is a philosophical question concerning the significance of living or existence in general.

{context}

Question: {question}
Answer:
"""

def rag(
    model: gguf.Model,
    tokenizer: gguf.GGUFTokenizer,
    prompt: str,
    max_tokens: int,
    temp: float = 0.0,
):
    prompt = tokenizer.encode(prompt)
    
    tic = time.time()
    tokens = []
    skip = 0
    for token, n in zip(
        gguf.generate(prompt, model, args.temp),
        range(args.max_tokens),
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
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument(
        "--gguf",
        type=str,
        help="The GGUF file to load (and optionally download).",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="The Hugging Face repo if downloading from the Hub.",
    )
    parser.add_argument(
        "--vdb",
        type=str,
        default="vdb.npz",
        help="The path to read the vector DB",
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
        default=612,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp",
        help="The sampling temperature.",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--question",
        help="The question that needs to be answered",
        default="what is flash attention?",
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")

    args = parser.parse_args()
    mx.random.seed(args.seed)
    m = VectorDB(args.vdb)
    context = m.query(args.question)
    args.question = TEMPLATE.format(context=context, question=args.question)
    model, tokenizer = gguf.load("tinyllama/tinyllama-1.1b-chat-v1.0.Q4_0.gguf")
    rag(model, tokenizer, args.question, args.max_tokens, args.temp)
