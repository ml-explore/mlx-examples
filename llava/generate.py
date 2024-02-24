import argparse
import os

import mlx.core as mx
import mlx.nn as nn
import requests
from PIL import Image
from transformers import AutoProcessor

from llava import LlavaModel


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate text from an image using a model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/llava-hf/llava-1.5-7b-hf",
        help="Path to the model directory.",
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
        help="Prompt to use for the model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3, help="Temperature for sampling."
    )
    return parser.parse_args()


def load_image(image_source):
    if image_source.startswith(("http://", "https://")):
        try:
            response = requests.get(image_source, stream=True)
            response.raise_for_status()
            return Image.open(response.raw)
        except requests.HTTPError as e:
            print(f"Failed to load image from URL: {e}")
            return None
    elif os.path.isfile(image_source):
        try:
            return Image.open(image_source)
        except IOError as e:
            print(f"Failed to load image from path: {e}")
            return None
    else:
        print("The image source is neither a valid URL nor a file path.")
        return None


def initialize_model(model_path):
    processor = AutoProcessor.from_pretrained(model_path)
    model = LlavaModel.from_pretrained(model_path)
    return processor, model


def prepare_inputs(processor, image, prompt):
    inputs = processor(prompt, image, return_tensors="np")
    pixel_values = mx.array(inputs["pixel_values"])
    input_ids = mx.array(inputs["input_ids"])
    return input_ids, pixel_values


def sample(logits, temperature=0.0):
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    else:
        return mx.random.categorical(logits * (1 / temperature))


def generate_text(input_ids, pixel_values, model, processor, max_tokens, temperature):
    input_embeds = model.get_input_embeddings(input_ids, pixel_values)
    logits, cache = model.language_model(
        input_ids, cache=None, inputs_embeds=input_embeds
    )
    logits = logits[:, -1, :]
    y = sample(logits, temperature=temperature)
    tokens = [y.item()]

    for _ in range(max_tokens):
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
    raw_image = load_image(args.image)
    if raw_image is None:
        return

    processor, model = initialize_model(args.model)
    input_ids, pixel_values = prepare_inputs(processor, raw_image, args.prompt)
    print(args.prompt)
    generated_text = generate_text(
        input_ids, pixel_values, model, processor, args.max_tokens, args.temperature
    )
    print(generated_text)


if __name__ == "__main__":
    main()
