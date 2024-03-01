# Copyright © 2024 Apple Inc.

import argparse
import codecs

import mlx.core as mx
from transformers import AutoProcessor
from utils import get_model_path, load_image, prepare_inputs

from llava import LlavaModel


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate text from an image using a model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        help="URL or path of the image to process.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="USER: <image>\nWhat are these?\nASSISTANT:",
        help="Message to be processed by the model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temp", type=float, default=0.3, help="Temperature for sampling."
    )
    return parser.parse_args()


def load_model(model_path):
    processor = AutoProcessor.from_pretrained(model_path)

    model = LlavaModel.from_pretrained(get_model_path(model_path))
    return processor, model


def sample(logits, temperature=0.0):
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    else:
        return mx.random.categorical(logits * (1 / temperature))


def generate_text(input_ids, pixel_values, model, processor, max_tokens, temperature):

    logits, cache = model(input_ids, pixel_values)
    logits = logits[:, -1, :]
    y = sample(logits, temperature=temperature)
    tokens = [y.item()]

    for n in range(max_tokens - 1):
        logits, cache = model.language_model(y[None], cache=cache)
        logits = logits[:, -1, :]
        y = sample(logits, temperature)
        token = y.item()
        if token == processor.tokenizer.eos_token_id:
            break
        tokens.append(token)

    return processor.tokenizer.decode(tokens)


def main():
    args = parse_arguments()
    image = load_image(args.image)
    processor, model = load_model(args.model)

    prompt = codecs.decode(args.prompt, "unicode_escape")

    input_ids, pixel_values = prepare_inputs(processor, image, prompt)

    print(prompt)
    generated_text = generate_text(
        input_ids, pixel_values, model, processor, args.max_tokens, args.temp
    )
    print(generated_text)


if __name__ == "__main__":
    main()
