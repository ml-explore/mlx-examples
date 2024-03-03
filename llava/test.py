# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from llava import LlavaModel

MODEL_PATH = "llava-hf/llava-1.5-7b-hf"
PROMPT = "USER: <image>\nWhat are these?\nASSISTANT:"
IMAGE_FILE = "http://images.cocodataset.org/val2017/000000039769.jpg"


def load_mlx_models(path):
    model = LlavaModel.from_pretrained(path)
    model.eval()
    return model


def load_hf_models(path):
    model = LlavaForConditionalGeneration.from_pretrained(path)
    model.eval()
    return model


class TestVisionTower(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mx_llava = load_mlx_models(MODEL_PATH)
        cls.hf_llava = load_hf_models(MODEL_PATH)
        cls.proc = AutoProcessor.from_pretrained(MODEL_PATH)

    def test_image_features(self):
        raw_image = Image.open(requests.get(IMAGE_FILE, stream=True).raw)
        vision_feature_layer = -2
        with torch.no_grad():
            pixel_values = self.proc(PROMPT, raw_image, return_tensors="pt")[
                "pixel_values"
            ]

            hf_pixel_values = pixel_values
            mx_pixel_values = mx.array(pixel_values.numpy()).transpose(0, 2, 3, 1)

            _, _, hidden_states = self.mx_llava.vision_tower(
                mx_pixel_values,
                output_hidden_states=True,
            )

            mx_elected_image_feature = hidden_states[vision_feature_layer]
            mx_image_features = self.mx_llava.multi_modal_projector(
                mx_elected_image_feature
            )

            hf_image_outputs = self.hf_llava.vision_tower(
                hf_pixel_values, output_hidden_states=True
            )
            hf_elected_image_feature = hf_image_outputs.hidden_states[
                vision_feature_layer
            ]
            hf_image_features = self.hf_llava.multi_modal_projector(
                hf_elected_image_feature
            )

            self.assertTrue(
                mx.allclose(
                    mx_image_features,
                    mx.array(hf_image_features.numpy()),
                    atol=1e-2,
                )
            )


class TestLlava(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mx_llava = load_mlx_models(MODEL_PATH)
        cls.hf_llava = load_hf_models(MODEL_PATH)
        cls.proc = AutoProcessor.from_pretrained(MODEL_PATH)

    def test_merge_input_ids_with_image_features(self):
        raw_image = Image.open(requests.get(IMAGE_FILE, stream=True).raw)
        vision_feature_layer = -2
        with torch.no_grad():
            values = self.proc(PROMPT, raw_image, return_tensors="pt")
            pixel_values = values["pixel_values"]
            input_ids = values["input_ids"]

            hf_pixel_values = pixel_values
            mx_pixel_values = mx.array(pixel_values.numpy()).transpose(0, 2, 3, 1)

            _, _, hidden_states = self.mx_llava.vision_tower(
                mx_pixel_values,
                output_hidden_states=True,
            )
            mx_input_ids = mx.array(input_ids.numpy())
            mx_elected_image_feature = hidden_states[vision_feature_layer]
            mx_image_features = self.mx_llava.multi_modal_projector(
                mx_elected_image_feature
            )
            mx_inputs_embeds = self.mx_llava.language_model.model.embed_tokens(
                mx_input_ids
            )
            mx_final_embedding = self.mx_llava._merge_input_ids_with_image_features(
                mx_image_features, mx_inputs_embeds, mx_input_ids
            )

            hf_image_outputs = self.hf_llava.vision_tower(
                hf_pixel_values, output_hidden_states=True
            )
            hf_elected_image_feature = hf_image_outputs.hidden_states[
                vision_feature_layer
            ]
            hf_image_features = self.hf_llava.multi_modal_projector(
                hf_elected_image_feature
            )
            hf_inputs_embeds = self.hf_llava.get_input_embeddings()(input_ids)
            hf_final_embedding, _, _, _ = (
                self.hf_llava._merge_input_ids_with_image_features(
                    hf_image_features,
                    hf_inputs_embeds,
                    input_ids,
                    attention_mask=input_ids,
                    labels=torch.ones_like(input_ids),
                )
            )

            self.assertTrue(
                mx.allclose(
                    mx_final_embedding,
                    mx.array(hf_final_embedding.numpy()),
                    atol=1e-1,
                )
            )

    def test_generated_tokens(self):
        raw_image = Image.open(requests.get(IMAGE_FILE, stream=True).raw)
        with torch.no_grad():
            hf_inputs = self.proc(PROMPT, raw_image, return_tensors="pt")
            hf_outputs = self.hf_llava(**hf_inputs)
            hf_logits = hf_outputs.logits

            mx_inputs = self.proc(PROMPT, raw_image, return_tensors="np")
            pixel_values = mx.array(mx_inputs["pixel_values"])
            input_ids = mx.array(mx_inputs["input_ids"])

            mx_logits, _ = self.mx_llava(input_ids, pixel_values)

            self.assertTrue(
                mx.allclose(
                    mx_logits[:, -1, :].argmax(axis=-1),
                    mx.array(hf_logits.numpy())[:, -1, :].argmax(axis=-1),
                    atol=1e-2,
                )
            )


if __name__ == "__main__":
    unittest.main()
