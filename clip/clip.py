from typing import Tuple

from image_processor import CLIPImageProcessor
from model import CLIPModel
from tokenizer import CLIPTokenizer
from mlx.nn.losses import cosine_similarity_loss
import logging

logging.basicConfig(level=logging.INFO)

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

    # Get text embeddings:
    text_embeds = output.text_embeds
    image_embeds = output.image_embeds

    # Compute similarity scores for text embeddings:
    text_similarities = cosine_similarity_loss(text_embeds[0], text_embeds[1], axis=-1)

    # Compute similarity scores for image embeddings:
    image_similarities = cosine_similarity_loss(image_embeds[0], image_embeds[1], axis=-1)

    logging.info(f"Text embeddings shape: {text_embeds.shape}")
    logging.info(f"Similarity score between 'a photo of a cat' and 'a photo of a dog' (Text): {text_similarities}")
    logging.info(f"Similarity score between 'a photo of a cat' and 'a photo of a dog' (Image): {image_similarities}")
