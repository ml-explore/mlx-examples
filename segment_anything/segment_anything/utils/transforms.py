from copy import deepcopy
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image


class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched mlx tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(
            image.shape[0], image.shape[1], self.target_length
        )
        return np.array(
            Image.fromarray(image).resize(
                target_size[::-1], resample=Image.Resampling.BILINEAR
            )
        )

    def apply_coords(
        self, coords: mx.array, original_size: Tuple[int, ...]
    ) -> mx.array:
        """
        Expects a mlx tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        return coords * mx.array([new_w / old_w, new_h / old_h])

    def apply_boxes(self, boxes: mx.array, original_size: Tuple[int, ...]) -> mx.array:
        """
        Expects a mlx tensor with shape ...x4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(
        oldh: int, oldw: int, long_side_length: int
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
