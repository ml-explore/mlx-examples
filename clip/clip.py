from typing import Tuple

from image_processor import CLIPImageProcessor
from model import CLIPModel
from tokenizer import CLIPTokenizer


def load(model_dir: str) -> Tuple[CLIPModel, CLIPTokenizer, CLIPImageProcessor]:
    model = CLIPModel.from_pretrained(model_dir)
    tokenizer = CLIPTokenizer.from_pretrained(model_dir)
    img_processor = CLIPImageProcessor.from_pretrained(model_dir)
    return model, tokenizer, img_processor


if __name__ == "__main__":
    from PIL import Image

    model, tokenizer, img_processor = load("mlx_model")
    inputs = {
        "input_ids": tokenizer(["a photo of a cat", "a photo of a dog"]),
        "pixel_values": img_processor(
            [Image.open("assets/cat.jpeg"), Image.open("assets/dog.jpeg")]
        ),
    }
    output = model(**inputs)

    # Get text and image embeddings:
    text_embeds = output.text_embeds
    image_embeds = output.image_embeds
    print("Text embeddings shape:", text_embeds.shape)
    print("Image embeddings shape:", image_embeds.shape)
