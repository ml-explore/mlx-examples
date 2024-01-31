from typing import Tuple

from image_processor import CLIPImageProcessor
from model import CLIPModel
from tokenizer import CLIPTokenizer


def load(model_dir: str) -> Tuple[CLIPModel, CLIPTokenizer, CLIPImageProcessor]:
    model = CLIPModel.from_pretrained(model_dir)
    tokenizer = CLIPTokenizer.from_pretrained(model_dir)
    img_processor = CLIPImageProcessor.from_pretrained(model_dir)
    return model, tokenizer, img_processor
