# Copyright © 2023-2024 Apple Inc.

import json
from pathlib import Path
from typing import Any, List, Tuple, Union

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image


class CLIPImageProcessor:
    """
    A simple port of https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/image_processing_clip.py.
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

    def __call__(self, images: List[Any]) -> mx.array:
        return mx.concatenate(
            [self._preprocess(image)[None, :, :, :] for image in images], axis=0
        )

    def _preprocess(self, image: Image) -> mx.array:
        if self.do_resize:
            image = self._resize(image, self.size)
        if self.do_center_crop:
            image = self._center_crop(image, (self.crop_size, self.crop_size))
        image = self._to_mlx(image)
        image = self._rescale(image)
        if self.do_normalize:
            image = self._normalize(image, self.image_mean, self.image_std)
        return image

    def _resize(self, image: Image, short_size: int) -> Image:
        width, height = image.size
        # Specified size only for the smallest edge
        short, long = (width, height) if width <= height else (height, width)
        if short == short_size:
            return image
        # New sizes
        new_short = short_size
        new_long = int(short_size * long / short)
        if width <= height:
            short_size = (new_short, new_long)
        else:
            short_size = (new_long, new_short)
        return image.resize(short_size)

    def _center_crop(self, image: Image, size: Tuple[int, int]) -> Image:
        # We perform the crop in (C, H, W) format and then convert to the output format
        original_width, original_height = image.size
        crop_height, crop_width = size
        # In case size is odd, (image_shape[0] + size[0]) // 2 won't give the proper result.
        top = (original_height - crop_height) // 2
        bottom = top + crop_height
        # In case size is odd, (image_shape[1] + size[1]) // 2 won't give the proper result.
        left = (original_width - crop_width) // 2
        right = left + crop_width
        return image.crop((left, top, right, bottom))

    def _to_mlx(self, image: Image) -> mx.array:
        return mx.array(np.array(image))

    def _rescale(self, image: mx.array) -> mx.array:
        return image.astype(mx.float32) * (1 / 255.0)

    def _normalize(self, image: mx.array, mean: mx.array, std: mx.array) -> mx.array:
        return (image - mean) / std

    @staticmethod
    def from_pretrained(path: str):
        path = Path(path)
        config_file = path / "preprocessor_config.json"
        with open(config_file, encoding="utf-8") as f:
            config = json.load(f)
        return CLIPImageProcessor(**config)
