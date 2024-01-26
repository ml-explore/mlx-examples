from pathlib import Path
from typing import Any, List, Union

import mlx.core as mx
from transformers import CLIPImageProcessor as TFCLIPImageProcessor


class CLIPImageProcessor:
    def __init__(self, path: Union[Path, str]) -> None:
        self.tf_processor = TFCLIPImageProcessor.from_pretrained(path)

    def __call__(self, images: List[Any]) -> mx.array:
        return mx.array(
            self.tf_processor(images, data_format="channels_last", return_tensors="np")[
                "pixel_values"
            ]
        )

    @staticmethod
    def from_pretrained(path: Union[Path, str]):
        return CLIPImageProcessor(path)
