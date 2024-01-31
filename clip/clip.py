from pathlib import Path
from typing import Tuple

from image_processor import CLIPImageProcessor
from model import CLIPModel
from tokenizer import CLIPTokenizer


def load(path_or_hf_repo: str) -> Tuple[CLIPModel, CLIPTokenizer, CLIPImageProcessor]:
    model = CLIPModel.from_pretrained(Path(path_or_hf_repo))
    tokenizer = CLIPTokenizer.from_pretrained(Path(path_or_hf_repo))
    img_processor = CLIPImageProcessor.from_pretrained(Path(path_or_hf_repo))
    return (model, tokenizer, img_processor)
