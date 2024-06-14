# Copyright © 2024 Apple Inc.

import argparse
import codecs
from pathlib import Path

import mlx.core as mx
import requests
from PIL import Image
from transformers import AutoProcessor
from llava import LlavaModel
from typing import Tuple, Dict, Union

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from an image using a model.")
    parser.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf", help="The path to the local model directory or Hugging Face repo.")
    parser.add_argument("--image", type=str, default="http://images.cocodataset.org/val2017/000000039769.jpg", help="URL or path of the image to process.")
    parser.add_argument("--prompt", type=str, default="USER: <image>\nWhat are these?\nASSISTANT:", help="Message to be processed by the model.")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of tokens to generate.")
    parser.add_argument("--temp", type=float, default=0.3, help="Temperature for sampling.")
    parser.add_argument("--eos-token", type=str, default=None, help="End of sequence token for tokenizer")
    return parser.parse_args()

def load_image(image_source: str) -> Image.Image:
    """
    Load an image from a URL or local file.

    Args:
        image_source (str): URL or path to the image.

    Returns:
        Image.Image: Loaded image.

    Raises:
        ValueError: If the image cannot be loaded.
    """
    try:
        if image_source.startswith(("http://", "https://")):
            response = requests.get(image_source, stream=True)
            response.raise_for_status()
            return Image.open(response.raw)
        elif Path(image_source).is_file():
            return Image.open(image_source)
        else:
            raise ValueError("The image source must be a valid URL or existing file path.")
    except Exception as e:
        raise ValueError(f"Failed to load image from {image_source} with error: {e}")

def prepare_inputs(processor: AutoProcessor, image: Union[str, Image.Image], prompt: str) -> Tuple[mx.ndarray, mx.ndarray]:
    """
    Prepare inputs for the model.

    Args:
        processor (AutoProcessor): Processor for handling image and text inputs.
        image (Union[str, Image.Image]): Image source or PIL image.
        prompt (str): Text prompt to be processed.

    Returns:
        Tuple[mx.ndarray, mx.ndarray]: Processed input IDs and pixel values.
    """
    if isinstance(image, str):
        image = load_image(image)
    inputs = processor(prompt, image, return_tensors="np")
    pixel_values = mx.array(inputs["pixel_values"])
    input_ids = mx.array(inputs["input_ids"])
    return input_ids, pixel_values

def load_model(model_path: str, tokenizer_config: Dict[str, Union[str, None]]) -> Tuple[AutoProcessor, LlavaModel]:
    """
    Load the processor and model.

    Args:
        model_path (str): Path to the model.
        tokenizer_config (Dict[str, Union[str, None]]): Configuration for the tokenizer.

    Returns:
        Tuple[AutoProcessor, LlavaModel]: Loaded processor and model.
    """
    processor = AutoProcessor.from_pretrained(model_path, **tokenizer_config)
    model = LlavaModel.from_pretrained(model_path)
    return processor, model

def sample(logits: mx.ndarray, temperature: float = 0.0) -> mx.ndarray:
    """
    Sample from the logits using the specified temperature.

    Args:
        logits (mx.ndarray): Logits from the model.
        temperature (float): Sampling temperature.

    Returns:
        mx.ndarray: Sampled token IDs.
    """
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    else:
        return mx.random.categorical(logits * (1 / temperature))

def generate_text(input_ids: mx.ndarray, pixel_values: mx.ndarray, model: LlavaModel, processor: AutoProcessor, max_tokens: int, temperature: float) -> str:
    """
    Generate text from model inputs.

    Args:
        input_ids (mx.ndarray): Input IDs.
        pixel_values (mx.ndarray): Pixel values.
        model (LlavaModel): Loaded model.
        processor (AutoProcessor): Processor for handling inputs.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.

    Returns:
        str: Generated text.
    """
    logits, cache = model(input_ids, pixel_values)
    logits = logits[:, -1, :]
    y = sample(logits, temperature=temperature)
    tokens = [y.item()]

    for _ in range(max_tokens - 1):
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

    tokenizer_config = {"eos_token": args.eos_token} if args.eos_token else {}

    processor, model = load_model(args.model, tokenizer_config)

    prompt = codecs.decode(args.prompt, "unicode_escape")

    input_ids, pixel_values = prepare_inputs(processor, args.image, prompt)

    print(prompt)
    generated_text = generate_text(input_ids, pixel_values, model, processor, args.max_tokens, args.temp)
    print(generated_text)

if __name__ == "__main__":
    main()
