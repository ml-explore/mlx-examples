import unittest

import mlx.core as mx
import numpy as np
import requests
import torch
from PIL import Image
from processing_llava import LlavaProcessor
from transformers import AutoProcessor, LlavaForConditionalGeneration

MLX_PATH = "models/llava-hf/llava-1.5-7b-hf"
HF_PATH = "models/llava-hf/llava-1.5-7b-hf"


def load_mlx_models(path):
    processor = LlavaProcessor()
    return processor, None


def load_hf_models(path):
    processor = AutoProcessor.from_pretrained(path)
    model = LlavaForConditionalGeneration.from_pretrained(path)

    return processor, model


class TestCLIP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mx_proc, cls.mx_llava = load_mlx_models(MLX_PATH)
        cls.hf_proc, cls.hf_llava = load_hf_models(HF_PATH)

    def test_processor(self):
        prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
        image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
        raw_image = Image.open(requests.get(image_file, stream=True).raw)

        hf_data = mx.array(
            np.array(
                self.hf_proc(prompt, raw_image, return_tensors="pt")["pixel_values"]
            )
        ).transpose(0, 2, 3, 1)

        mx_data = self.mx_proc(prompt, [raw_image])["pixel_values"]

        self.assertTrue(mx.allclose(mx_data, hf_data, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
