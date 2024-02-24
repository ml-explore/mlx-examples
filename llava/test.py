import unittest

import mlx.core as mx
import numpy as np
import requests
from PIL import Image
from processing_llava import LlavaProcessor
from transformers import AutoProcessor, LlavaForConditionalGeneration

from llava import LlavaModel

MODEL_PATH = "models/llava-hf/llava-1.5-7b-hf"


def load_mlx_models(path):
    model = LlavaModel.from_pretrained(path)
    return model


def load_hf_models(path):
    model = LlavaForConditionalGeneration.from_pretrained(path)
    return model


class TestCLIP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mx_llava = load_mlx_models(MODEL_PATH)
        cls.hf_llava = load_hf_models(MODEL_PATH)
        cls.proc = AutoProcessor.from_pretrained(MODEL_PATH)

    def test_processor(self):
        prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
        image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
        raw_image = Image.open(requests.get(image_file, stream=True).raw)

        hf_data = mx.array(
            self.proc(prompt, raw_image, return_tensors="np")["pixel_values"]
        )

        mx_data = mx.array(
            self.proc(prompt, raw_image, return_tensors="np")["pixel_values"]
        )

        self.assertTrue(mx.allclose(mx_data, hf_data, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
