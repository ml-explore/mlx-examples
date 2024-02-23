import unittest

import mlx.core as mx
import model
import numpy as np
import torch
import transformers
from image_processor import CLIPImageProcessor
from PIL import Image
from tokenizer import CLIPTokenizer
from transformers import AutoTokenizer
from transformers.image_processing_utils import ChannelDimension

MLX_PATH = "mlx_model"
HF_PATH = "openai/clip-vit-base-patch32"


def load_mlx_models(path):
    image_proc = CLIPImageProcessor.from_pretrained(path)
    tokenizer = CLIPTokenizer.from_pretrained(path)
    clip = model.CLIPModel.from_pretrained(path)
    return image_proc, tokenizer, clip


def load_hf_models(path):
    image_proc = transformers.CLIPImageProcessor.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    clip = transformers.CLIPModel.from_pretrained(path)
    return image_proc, tokenizer, clip


class TestCLIP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mx_image_proc, cls.mx_tokenizer, cls.mx_clip = load_mlx_models(MLX_PATH)
        cls.hf_image_proc, cls.hf_tokenizer, cls.hf_clip = load_hf_models(HF_PATH)

    def test_image_processor(self):
        image = Image.open("assets/cat.jpeg")

        mx_data = self.mx_image_proc([image])
        hf_data = mx.array(
            np.array(
                self.hf_image_proc([image], data_format=ChannelDimension.LAST)[
                    "pixel_values"
                ]
            )
        )
        self.assertTrue(mx.allclose(mx_data, hf_data, atol=1e-5))

    def test_text_tokenizer(self):
        texts = ["a photo of a cat", "a photo of a dog"]
        for txt in texts:
            self.assertTrue(
                np.array_equal(
                    self.mx_tokenizer.tokenize(txt)[None, :],
                    self.hf_tokenizer(txt, return_tensors="np")["input_ids"],
                ),
            )

    def test_text_encoder(self):
        texts = ["a photo of a cat", "a photo of a dog"]
        # Tokenize
        hf_tokens = self.hf_tokenizer(texts, return_tensors="pt")
        mx_tokens = self.mx_tokenizer(texts)
        # Get expected
        with torch.inference_mode():
            expected_out = self.hf_clip.text_model(**hf_tokens)
            expected_last_hidden = expected_out.last_hidden_state.numpy()
            expected_pooler_output = expected_out.pooler_output.numpy()
        out = self.mx_clip.text_model(mx_tokens)
        self.assertTrue(
            np.allclose(out.last_hidden_state, expected_last_hidden, atol=1e-5)
        )
        self.assertTrue(
            np.allclose(out.pooler_output, expected_pooler_output, atol=1e-5)
        )

    def test_vision_encoder(self):
        # Load and process test image
        x = self.hf_image_proc(
            images=[Image.open("assets/dog.jpeg")], return_tensors="np"
        ).pixel_values

        # Infer with HuggingFace model
        with torch.inference_mode():
            # Get expected
            x_tc = torch.tensor(x)
            expected_out = self.hf_clip.vision_model(x_tc, output_hidden_states=True)
            expected_last_hidden = expected_out.last_hidden_state.numpy()
            expected_pooler_output = expected_out.pooler_output.numpy()
            expected_hidden_states = [hs.numpy() for hs in expected_out.hidden_states]
        # Test MLX vision encoder
        out = self.mx_clip.vision_model(
            mx.array(x.transpose(0, 2, 3, 1)), output_hidden_states=True
        )
        self.assertTrue(
            np.allclose(
                out.last_hidden_state, expected_last_hidden, rtol=1e-4, atol=1e-3
            ),
        )
        self.assertTrue(
            np.allclose(
                out.pooler_output, expected_pooler_output, rtol=1e-4, atol=1e-3
            ),
        )
        for expected_hs, out_hs in zip(expected_hidden_states, out.hidden_states):
            self.assertTrue(
                np.allclose(expected_hs, out_hs, rtol=1e-4, atol=1e-3),
            )

    def test_clip_model(self):
        image_input = self.hf_image_proc(
            images=[Image.open("assets/cat.jpeg"), Image.open("assets/dog.jpeg")],
            return_tensors="np",
        )["pixel_values"]
        text = ["a photo of a cat", "a photo of a dog"]
        tokens = self.hf_tokenizer(text, return_tensors="np")["input_ids"]
        with torch.inference_mode():
            expected_out = self.hf_clip(
                input_ids=torch.tensor(tokens),
                pixel_values=torch.tensor(image_input),
                return_loss=True,
            )

        out = self.mx_clip(
            input_ids=mx.array(tokens),
            pixel_values=mx.array(image_input.transpose((0, 2, 3, 1))),
            return_loss=True,
        )

        self.assertTrue(
            np.allclose(out.text_embeds, expected_out.text_embeds, atol=1e-5)
        )
        self.assertTrue(
            np.allclose(out.image_embeds, expected_out.image_embeds, atol=1e-5)
        )
        self.assertTrue(np.allclose(out.loss, expected_out.loss, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
