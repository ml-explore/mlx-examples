# Copyright © 2023-2024 Apple Inc.

import json
from pathlib import Path
from typing import List, Tuple

import mlx.core as mx
import numpy as np
from PIL.Image import Image


class CLIPImageProcessor:
    """
    A simple port of
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/image_processing_clip.py.
    """

    def __init__(
        self,
        crop_size: int = 224,
        do_center_crop: bool = True,
        do_normalize: bool = True,
        do_resize: bool = True,
        image_mean: List[float] = [0.48145466, 0.4578275, 0.40821073],
        image_std: List[float] = [0.26862954, 0.26130258, 0.27577711],
        size: int = 224,
        **kwargs
    ) -> None:
        self.crop_size = crop_size
        self.do_center_crop = do_center_crop
        self.do_normalize = do_normalize
        self.do_resize = do_resize
        self.image_mean = mx.array(image_mean)
        self.image_std = mx.array(image_std)
        self.size = size

    def __call__(self, images: List[Image]) -> mx.array:
        return mx.concatenate(
            [self._preprocess(image)[None] for image in images], axis=0
        )

    def _preprocess(self, image: Image) -> mx.array:
        if self.do_resize:
            image = resize(image, self.size)
        if self.do_center_crop:
            image = center_crop(image, (self.crop_size, self.crop_size))
        image = mx.array(np.array(image))
        image = rescale(image)
        if self.do_normalize:
            image = normalize(image, self.image_mean, self.image_std)
        return image

    @staticmethod
    def from_pretrained(path: str):
        path = Path(path)
        with open(path / "preprocessor_config.json", encoding="utf-8") as f:
            config = json.load(f)
        return CLIPImageProcessor(**config)


def resize(image: Image, short_size: int) -> Image:
    """
    Resize so small size to short_size
    """
    width, height = image.size
    short = min(width, height)
    long = max(width, height)
    if short == short_size:
        return image
    new_short = short_size
    new_long = int(short_size * long / short)
    new_size = (new_short, new_long) if width <= height else (new_long, new_short)
    return image.resize(new_size)


def center_crop(image: Image, size: Tuple[int, int]) -> Image:
    if size[0] % 2 != 0 or size[1] % 2 != 0:
        raise ValueError("Only even crop sizes supported.")
    original_width, original_height = image.size
    crop_height, crop_width = size
    top = (original_height - crop_height) // 2
    bottom = top + crop_height
    left = (original_width - crop_width) // 2
    right = left + crop_width
    return image.crop((left, top, right, bottom))


def rescale(image: mx.array) -> mx.array:
    return image.astype(mx.float32) * (1 / 255.0)


def normalize(image: mx.array, mean: mx.array, std: mx.array) -> mx.array:
    return (image - mean) / std
