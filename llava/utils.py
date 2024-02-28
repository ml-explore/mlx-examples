from pathlib import Path
from huggingface_hub import snapshot_download
import os
import requests
from PIL import Image
import mlx.core as mx


def get_model_path(path_or_hf_repo: str) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.

    Returns:
        Path: The path to the model.
    """
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.py",
                    "tokenizer.model",
                    "*.tiktoken",
                ],
            )
        )
    return model_path


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


def prepare_inputs(processor, image, prompt):
    inputs = processor(prompt, image, return_tensors="np")
    pixel_values = mx.array(inputs["pixel_values"])
    input_ids = mx.array(inputs["input_ids"])
    return input_ids, pixel_values
